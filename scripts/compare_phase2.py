from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from visionguard.config import load_config_from_env
from visionguard.data import load_test_items
from visionguard.model import PatchCoreModel
from visionguard.visualization import generate_heatmap, overlay_heatmap

EXPERIMENT_ID = "patchcore_topk_mean_1pct_v1"
METRICS = (
    "image_auroc",
    "image_average_precision",
    "image_precision",
    "image_recall",
    "image_f1",
    "pixel_auroc",
    "pixel_average_precision",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "inference_seconds_per_image",
)
MARKER = "## Phase 2 controlled experiment"


def _macro(rows: list[dict]) -> dict:
    output = {}
    for metric in METRICS:
        values = [row.get(metric) for row in rows if row.get(metric) is not None]
        output[metric] = sum(values) / len(values) if values else None
    return output


def _save_baseline_examples(
    category: str,
    predictions: list[dict],
    model_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> list[dict]:
    selected: dict[str, dict] = {}
    for prediction in predictions:
        label = int(prediction["label"])
        predicted = int(prediction["prediction"])
        kind = (
            "true_positive"
            if label == 1 and predicted == 1
            else "false_positive"
            if label == 0 and predicted == 1
            else "false_negative"
            if label == 1 and predicted == 0
            else None
        )
        if kind is not None and kind not in selected:
            selected[kind] = prediction
    if not selected:
        return []

    items = {str(item.image_path): item for item in load_test_items(dataset_root, category)}
    model = PatchCoreModel.load(model_path, device="cpu")
    examples = []
    for kind, prediction in selected.items():
        item = items[prediction["image_path"]]
        score, raw_map = model.predict_image(item.image_path)
        target = output_dir / category / kind
        target.mkdir(parents=True, exist_ok=True)
        original_path = target / "original.png"
        heatmap_path = target / "heatmap.png"
        overlay_path = target / "overlay.png"
        raw_scores_path = target / "raw_scores.npy"
        mask_path = target / "ground_truth.png" if item.mask is not None else None
        item.image.save(original_path)
        Image.fromarray(generate_heatmap(raw_map, item.image.size)).save(heatmap_path)
        Image.fromarray(overlay_heatmap(item.image, raw_map)).save(overlay_path)
        np.save(raw_scores_path, raw_map)
        if mask_path is not None:
            Image.fromarray(item.mask).save(mask_path)
        example = {
            "kind": kind,
            "split": "official test",
            "image_path": str(item.image_path),
            "ground_truth_mask": str(mask_path) if mask_path else None,
            "image_score": score,
            "threshold": prediction["threshold"],
            "label": prediction["label"],
            "prediction": prediction["prediction"],
            "original": str(original_path),
            "heatmap": str(heatmap_path),
            "overlay": str(overlay_path),
            "raw_anomaly_scores": str(raw_scores_path),
        }
        (target / "metadata.json").write_text(
            json.dumps(example, indent=2) + "\n", encoding="utf-8"
        )
        examples.append(example)
    return examples


def _write_comparison(
    baseline_path: Path,
    candidate_root: Path,
    dataset_root: Path,
    reports_root: Path,
) -> dict:
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_path = candidate_root / "evaluation_results.json"
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline_rows = {row["category"]: row for row in baseline_payload["categories"]}
    candidate_rows = {
        row["category"]: row for row in candidate_payload["categories"]
    }
    categories = [
        category
        for category in ("candle", "macaroni1", "macaroni2")
        if category in baseline_rows and category in candidate_rows
    ]
    if not categories:
        raise ValueError("No shared baseline/candidate categories were found.")

    baseline_snapshot = reports_root / "experiments" / "baseline_patchcore"
    baseline_snapshot.mkdir(parents=True, exist_ok=True)
    for filename in (
        "evaluation_results.json",
        "per_category_metrics.csv",
        "image_predictions.csv",
        "evaluation_report.md",
        "benchmark_results.json",
    ):
        source = reports_root / filename
        destination = baseline_snapshot / filename
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)

    comparison_rows = []
    for category in categories:
        baseline = baseline_rows[category]
        candidate = candidate_rows[category]
        differences = {
            metric: (
                candidate.get(metric) - baseline.get(metric)
                if candidate.get(metric) is not None
                and baseline.get(metric) is not None
                else None
            )
            for metric in METRICS
        }
        comparison_rows.append(
            {
                "category": category,
                "baseline": {metric: baseline.get(metric) for metric in METRICS},
                "candidate": {metric: candidate.get(metric) for metric in METRICS},
                "delta_candidate_minus_baseline": differences,
                "baseline_threshold": baseline.get("image_threshold"),
                "candidate_threshold": candidate.get("image_threshold"),
                "baseline_confusion_matrix": baseline.get("image_confusion_matrix"),
                "candidate_confusion_matrix": candidate.get("image_confusion_matrix"),
            }
        )
        _save_baseline_examples(
            category,
            baseline["image_predictions"],
            Path("artifacts") / category / "model.npz",
            dataset_root,
            reports_root / "experiments" / "baseline_patchcore",
        )
        candidate_artifact = Path(candidate["candidate_model_artifact"])
        with np.load(candidate_artifact, allow_pickle=False) as artifact:
            artifact_metadata = json.loads(str(artifact["metadata"].item()))
            candidate["memory_bank_bytes"] = int(artifact["memory_bank"].nbytes)
        candidate["model_artifact_bytes"] = candidate_artifact.stat().st_size
        saved_config = artifact_metadata["config"]
        config = {
            "experiment_id": EXPERIMENT_ID,
            "model_version": "baseline-fitted-memory-bank+topk-mean-1pct",
            "category": category,
            "dataset_type": "VisA",
            "dataset_identity": str(dataset_root.resolve()),
            "training_split": "official train/good; baseline memory bank reused",
            "calibration_split": "official train/good normal images only",
            "evaluation_split": "official test",
            "image_score_top_k_fraction": 0.01,
            "image_threshold_quantile": 0.99,
            "pixel_threshold_source": (
                "baseline calibration on normal train/good maps; spatial maps "
                "are unchanged by image aggregation"
            ),
            "random_seed": artifact_metadata["random_seed"],
            "image_size": artifact_metadata["input_size"],
            "sample_ratio": artifact_metadata["sample_ratio"],
            "image_score_aggregation": saved_config["image_score_aggregation"],
            "image_score_top_k_fraction": saved_config[
                "image_score_top_k_fraction"
            ],
            "candidate_model_artifact": candidate.get("candidate_model_artifact"),
        }
        config_path = candidate_root / category / "experiment_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        candidate["experiment_config"] = str(config_path)
        (candidate_root / category / "evaluation.json").write_text(
            json.dumps(candidate, indent=2) + "\n", encoding="utf-8"
        )
        from mlflow.tracking import MlflowClient

        tracking_uri = load_config_from_env().mlflow_tracking_uri
        if tracking_uri:
            client = MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name("visionguard-visa")
            if experiment is not None:
                run_name = f"phase2-{EXPERIMENT_ID}-{category}"
                matching = [
                    run
                    for run in client.search_runs(
                        [experiment.experiment_id], max_results=1000
                    )
                    if run.data.tags.get("mlflow.runName") == run_name
                ]
                for run in matching:
                    client.log_artifact(run.info.run_id, str(config_path))
                    client.log_param(
                        run.info.run_id,
                        "model_artifact_bytes",
                        candidate["model_artifact_bytes"],
                    )
                    client.log_param(
                        run.info.run_id,
                        "memory_bank_bytes",
                        candidate["memory_bank_bytes"],
                    )

    baseline_macro = _macro([baseline_rows[category] for category in categories])
    candidate_macro = _macro([candidate_rows[category] for category in categories])
    macro_delta = {
        metric: (
            candidate_macro[metric] - baseline_macro[metric]
            if candidate_macro[metric] is not None
            and baseline_macro[metric] is not None
            else None
        )
        for metric in METRICS
    }
    baseline_benchmark_path = reports_root / "experiments" / "phase2_baseline_benchmark.json"
    candidate_benchmark_path = candidate_root / "benchmark_results.json"
    benchmark_comparison = None
    if baseline_benchmark_path.is_file() and candidate_benchmark_path.is_file():
        baseline_benchmark = json.loads(
            baseline_benchmark_path.read_text(encoding="utf-8")
        )
        candidate_benchmark = json.loads(
            candidate_benchmark_path.read_text(encoding="utf-8")
        )
        benchmark_comparison = {
            "baseline": {
                "image_score_aggregation": baseline_benchmark.get(
                    "image_score_aggregation"
                ),
                "model_load_seconds": baseline_benchmark.get("model_load_seconds"),
                "end_to_end_prediction": baseline_benchmark["timings"][
                    "end_to_end_prediction"
                ],
                "throughput_images_per_second": baseline_benchmark.get(
                    "throughput_images_per_second"
                ),
                "image_count": baseline_benchmark.get("image_count"),
                "warmup_count": baseline_benchmark.get("warmup_count"),
                "device": baseline_benchmark.get("device"),
                "cpu_model": baseline_benchmark.get("cpu_model"),
                "software": baseline_benchmark.get("software"),
                "input_dimensions": baseline_benchmark.get("input_dimensions"),
                "cpu_peak_working_set_bytes": baseline_benchmark.get(
                    "cpu_peak_working_set_bytes"
                ),
                "model_artifact_bytes": baseline_benchmark.get(
                    "model_artifact_bytes"
                ),
                "memory_bank_bytes": baseline_benchmark.get("memory_bank_bytes"),
            },
            "candidate": {
                "image_score_aggregation": candidate_benchmark.get(
                    "image_score_aggregation"
                ),
                "model_load_seconds": candidate_benchmark.get("model_load_seconds"),
                "end_to_end_prediction": candidate_benchmark["timings"][
                    "end_to_end_prediction"
                ],
                "throughput_images_per_second": candidate_benchmark.get(
                    "throughput_images_per_second"
                ),
                "image_count": candidate_benchmark.get("image_count"),
                "warmup_count": candidate_benchmark.get("warmup_count"),
                "device": candidate_benchmark.get("device"),
                "cpu_model": candidate_benchmark.get("cpu_model"),
                "software": candidate_benchmark.get("software"),
                "input_dimensions": candidate_benchmark.get("input_dimensions"),
                "cpu_peak_working_set_bytes": candidate_benchmark.get(
                    "cpu_peak_working_set_bytes"
                ),
                "model_artifact_bytes": candidate_benchmark.get(
                    "model_artifact_bytes"
                ),
                "memory_bank_bytes": candidate_benchmark.get("memory_bank_bytes"),
            },
        }
    comparison = {
        "experiment_id": EXPERIMENT_ID,
        "protocol": (
            "Fixed top-1% mean aggregation; candidate reuses the baseline fitted "
            "memory bank. Candidate image threshold is recalibrated on normal "
            "train/good images at the baseline 99th percentile. Pixel map and "
            "pixel threshold are unchanged. Test split used only for final "
            "measurement; no test-based selection."
        ),
        "categories": comparison_rows,
        "benchmark": benchmark_comparison,
        "macro_average": {
            "baseline": baseline_macro,
            "candidate": candidate_macro,
            "delta_candidate_minus_baseline": macro_delta,
        },
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    json_path = reports_root / "phase2_comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    csv_path = reports_root / "phase2_comparison.csv"
    columns = ["category"]
    columns.extend(f"baseline_{metric}" for metric in METRICS)
    columns.extend(f"candidate_{metric}" for metric in METRICS)
    columns.extend(f"delta_{metric}" for metric in METRICS)
    columns.extend(
        (
            "baseline_threshold",
            "candidate_threshold",
            "baseline_confusion_matrix",
            "candidate_confusion_matrix",
        )
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in comparison_rows:
            flattened = {"category": row["category"]}
            flattened.update(
                {f"baseline_{key}": value for key, value in row["baseline"].items()}
            )
            flattened.update(
                {f"candidate_{key}": value for key, value in row["candidate"].items()}
            )
            flattened.update(
                {
                    f"delta_{key}": value
                    for key, value in row["delta_candidate_minus_baseline"].items()
                }
            )
            flattened["baseline_threshold"] = row["baseline_threshold"]
            flattened["candidate_threshold"] = row["candidate_threshold"]
            flattened["baseline_confusion_matrix"] = json.dumps(
                row["baseline_confusion_matrix"]
            )
            flattened["candidate_confusion_matrix"] = json.dumps(
                row["candidate_confusion_matrix"]
            )
            writer.writerow(flattened)

    lines = [
        MARKER,
        "",
        f"- Experiment: `{EXPERIMENT_ID}`",
        "- Candidate: mean of the highest 1% of spatial patch distances (8 of 784 patches at 224x224).",
        "- The fitted baseline memory bank is reused. Candidate image thresholds use only normal `train/good` scores at the 99th percentile; the pixel map and baseline normal-only pixel threshold are unchanged.",
        "- Test labels/masks are used only for these final measurements, not for choosing the candidate or calibrating thresholds.",
        "",
        "| Category | Image AUROC baseline → candidate | Image AP baseline → candidate | Image recall baseline → candidate | Image F1 baseline → candidate | Pixel AUROC baseline → candidate | Pixel AP baseline → candidate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        before, after = row["baseline"], row["candidate"]
        lines.append(
            f"| {row['category']} | {before['image_auroc']:.4f} → {after['image_auroc']:.4f} "
            f"| {before['image_average_precision']:.4f} → {after['image_average_precision']:.4f} "
            f"| {before['image_recall']:.4f} → {after['image_recall']:.4f} "
            f"| {before['image_f1']:.4f} → {after['image_f1']:.4f} "
            f"| {before['pixel_auroc']:.4f} → {after['pixel_auroc']:.4f} "
            f"| {before['pixel_average_precision']:.4f} → {after['pixel_average_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Three-category macro average",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        before = baseline_macro[metric]
        after = candidate_macro[metric]
        delta = macro_delta[metric]
        if before is not None and after is not None:
            lines.append(f"| {metric} | {before:.6f} | {after:.6f} | {delta:+.6f} |")
    if benchmark_comparison:
        baseline_result = benchmark_comparison["baseline"]
        candidate_result = benchmark_comparison["candidate"]
        baseline_timing = benchmark_comparison["baseline"][
            "end_to_end_prediction"
        ]
        candidate_timing = benchmark_comparison["candidate"][
            "end_to_end_prediction"
        ]
        lines.extend(
            [
                "",
                "### CPU benchmark comparison",
                "",
                f"- Device: `{baseline_result['device']}` on `{baseline_result['cpu_model']}`; both runs used {baseline_result['warmup_count']} warmups and {baseline_result['image_count']} measured candle test images at {baseline_result['input_dimensions'][0]}x{baseline_result['input_dimensions'][1]} input. Candidate ran on the same hardware/software configuration.",
                f"- Software (baseline): Python {baseline_result['software']['python']}, PyTorch {baseline_result['software']['torch']}, NumPy {baseline_result['software']['numpy']}.",
                "",
                "| Metric | Baseline max | Candidate top-1% mean |",
                "|---|---:|---:|",
                (
                    f"| Model load (s) | {baseline_result['model_load_seconds']:.3f} "
                    f"| {candidate_result['model_load_seconds']:.3f} |"
                ),
                (
                    f"| Mean prediction latency (ms) | {baseline_timing['mean_ms']:.3f} "
                    f"| {candidate_timing['mean_ms']:.3f} |"
                ),
                (
                    f"| Median prediction latency (ms) | {baseline_timing['median_ms']:.3f} "
                    f"| {candidate_timing['median_ms']:.3f} |"
                ),
                (
                    f"| P95 prediction latency (ms) | {baseline_timing['p95_ms']:.3f} "
                    f"| {candidate_timing['p95_ms']:.3f} |"
                ),
                (
                    f"| Throughput (images/s) | "
                    f"{benchmark_comparison['baseline']['throughput_images_per_second']:.3f} "
                    f"| {benchmark_comparison['candidate']['throughput_images_per_second']:.3f} |"
                ),
                (
                    f"| Peak process working set (bytes) | "
                    f"{benchmark_comparison['baseline']['cpu_peak_working_set_bytes']} "
                    f"| {benchmark_comparison['candidate']['cpu_peak_working_set_bytes']} |"
                ),
                (
                    f"| Model artifact size (bytes) | {baseline_result['model_artifact_bytes']} "
                    f"| {candidate_result['model_artifact_bytes']} |"
                ),
                (
                    f"| Memory bank size (bytes) | {baseline_result['memory_bank_bytes']} "
                    f"| {candidate_result['memory_bank_bytes']} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            f"Machine-readable comparison: `{json_path}` and `{csv_path}`.",
            "Diagnostic examples are in `reports/experiments/baseline_patchcore/` and the candidate experiment's `examples/` folders. True positives, false positives, and false negatives are labeled by the thresholded prediction; a false negative is not a detection.",
        ]
    )
    report_path = reports_root / "evaluation_report.md"
    existing = report_path.read_text(encoding="utf-8")
    if MARKER in existing:
        existing = existing.split(MARKER, maxsplit=1)[0].rstrip()
    report_path.write_text(
        existing.rstrip() + "\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase 2 candidate results with the preserved baseline."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("reports/evaluation_results.json"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("reports/experiments") / EXPERIMENT_ID,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/visa_prepared/1cls"),
    )
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    args = parser.parse_args()
    result = _write_comparison(
        args.baseline,
        args.candidate_root,
        args.dataset_root,
        args.reports_root,
    )
    print(json.dumps(result["macro_average"], indent=2))


if __name__ == "__main__":
    main()
