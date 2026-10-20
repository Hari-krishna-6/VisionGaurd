from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from scripts.experiment_score_aggregation import _save_diagnostic_examples
from visionguard.config import load_config_from_env
from visionguard.data import load_train_images, validate_dataset
from visionguard.evaluation import (
    evaluate_category,
    macro_average,
    weighted_quantile,
)
from visionguard.features import nearest_resize_pixel_counts
from visionguard.mlflow_utils import log_experiment
from visionguard.model import PatchCoreModel

EXPERIMENT_ID = "phase3_localization_v1"
DEFAULT_CATEGORIES = ("candle", "macaroni1", "macaroni2")
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
)


def _determine_nearest_pixel_threshold(
    train_paths: list[Path],
    model: PatchCoreModel,
    quantile: float = 0.995,
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    if not train_paths:
        raise ValueError("Cannot calibrate pixel threshold without normal training images")
    score_chunks = []
    weight_chunks = []
    for path in train_paths:
        _, patch_map, source_shape = model.predict_patch_map(path)
        score_chunks.append(patch_map.reshape(-1))
        weight_chunks.append(
            nearest_resize_pixel_counts(patch_map.shape, source_shape)
        )
    return weighted_quantile(
        np.concatenate(score_chunks),
        np.concatenate(weight_chunks),
        quantile,
    )


def _save_localization_diagnostics(
    model: PatchCoreModel,
    result: dict,
    dataset_root: Path,
    category: str,
    output_dir: Path,
) -> list[dict]:
    examples = _save_diagnostic_examples(
        model, result, dataset_root, category, output_dir
    )
    for example in examples:
        if example["label"] != 0 or example["ground_truth_mask"] is not None:
            continue
        with Image.open(example["image_path"]) as original:
            mask = np.zeros((original.height, original.width), dtype=np.uint8)
        mask_path = Path(example["original"]).parent / "ground_truth.png"
        Image.fromarray(mask).save(mask_path)
        example["ground_truth_mask"] = str(mask_path)
        metadata_path = Path(example["original"]).parent / "metadata.json"
        metadata_path.write_text(
            json.dumps(example, indent=2) + "\n", encoding="utf-8"
        )
    return examples


def _benchmark(
    model_path: Path,
    dataset_root: Path,
    category: str,
    output_path: Path,
    warmup: int,
    samples: int,
    run_name: str,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark",
        "--model",
        str(model_path),
        "--dataset-root",
        str(dataset_root),
        "--category",
        category,
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--output",
        str(output_path),
        "--run-name",
        run_name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Benchmark failed ({completed.returncode}): {completed.stderr[-4000:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Benchmark returned invalid JSON: {completed.stdout[-1000:]}"
        ) from exc


def _write_comparison(
    baseline_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    output_dir: Path,
) -> dict:
    categories = list(candidate_rows)
    comparison_rows = []
    for category in categories:
        baseline = baseline_rows[category]
        candidate = candidate_rows[category]
        comparison_rows.append(
            {
                "category": category,
                "baseline": {metric: baseline.get(metric) for metric in METRICS},
                "candidate": {metric: candidate.get(metric) for metric in METRICS},
                "delta_candidate_minus_baseline": {
                    metric: (
                        candidate.get(metric) - baseline.get(metric)
                        if candidate.get(metric) is not None
                        and baseline.get(metric) is not None
                        else None
                    )
                    for metric in METRICS
                },
                "baseline_threshold": baseline.get("image_threshold"),
                "candidate_threshold": candidate.get("image_threshold"),
                "baseline_pixel_threshold": baseline.get("pixel_threshold"),
                "candidate_pixel_threshold": candidate.get("pixel_threshold"),
                "baseline_benchmark": baseline["benchmark"],
                "candidate_benchmark": candidate["benchmark"],
                "baseline_stage_durations": baseline.get(
                    "phase3_stage_durations"
                ),
                "candidate_fit_seconds": None,
                "candidate_calibration_seconds": candidate[
                    "calibration_seconds"
                ],
                "candidate_evaluation_seconds": candidate[
                    "evaluation_seconds"
                ],
            }
        )

    baseline_macro = macro_average(
        [baseline_rows[category] for category in categories]
    )
    candidate_macro = macro_average(
        [candidate_rows[category] for category in categories]
    )
    comparison = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_configuration": {
            "feature_extractor": (
                "unchanged pretrained Wide-ResNet50-2 layer2/layer3 "
                "concatenation at the existing 28x28 feature grid"
            ),
            "anomaly_map_upsampling": "nearest",
            "baseline_anomaly_map_upsampling": "bilinear",
            "image_score_aggregation": "unchanged max over the same 784 patch distances",
            "memory_bank": "reused baseline bank fitted from official train/good only",
            "image_threshold": "baseline normal train/good calibration; unchanged",
            "pixel_threshold": (
                "candidate normal train/good maps at existing 99.5th percentile"
            ),
            "random_seed": 42,
        },
        "protocol": (
            "Candidate fixed before official test labels/masks were evaluated. "
            "Only the map upsampling interpolation changes; model features, "
            "patch distances, image scores, preprocessing, split, seed, and "
            "metric definitions remain unchanged. Candidate pixel threshold "
            "uses normal train/good maps only."
        ),
        "categories": comparison_rows,
        "macro_average": {
            "baseline": baseline_macro,
            "candidate": candidate_macro,
            "delta_candidate_minus_baseline": {
                metric: (
                    candidate_macro.get(metric) - baseline_macro.get(metric)
                    if candidate_macro.get(metric) is not None
                    and baseline_macro.get(metric) is not None
                    else None
                )
                for metric in METRICS
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )

    csv_columns = ["category"]
    csv_columns.extend(f"baseline_{metric}" for metric in METRICS)
    csv_columns.extend(f"candidate_{metric}" for metric in METRICS)
    csv_columns.extend(f"delta_{metric}" for metric in METRICS)
    csv_columns.extend(
        (
            "baseline_pixel_threshold",
            "candidate_pixel_threshold",
            "baseline_training_seconds",
            "baseline_calibration_seconds",
            "baseline_evaluation_seconds",
            "candidate_fit_seconds",
            "candidate_calibration_seconds",
            "candidate_evaluation_seconds",
        )
    )
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_columns)
        writer.writeheader()
        for row in comparison_rows:
            flattened = {"category": row["category"]}
            flattened.update(
                {f"baseline_{key}": value for key, value in row["baseline"].items()}
            )
            flattened.update(
                {
                    f"candidate_{key}": value
                    for key, value in row["candidate"].items()
                }
            )
            flattened.update(
                {
                    f"delta_{key}": value
                    for key, value in row[
                        "delta_candidate_minus_baseline"
                    ].items()
                }
            )
            flattened["baseline_pixel_threshold"] = row[
                "baseline_pixel_threshold"
            ]
            flattened["candidate_pixel_threshold"] = row[
                "candidate_pixel_threshold"
            ]
            baseline_durations = row["baseline_stage_durations"] or {}
            flattened["baseline_training_seconds"] = baseline_durations.get(
                "training_seconds"
            )
            flattened["baseline_calibration_seconds"] = baseline_durations.get(
                "calibration_seconds"
            )
            flattened["baseline_evaluation_seconds"] = baseline_durations.get(
                "evaluation_seconds"
            )
            flattened["candidate_fit_seconds"] = None
            flattened["candidate_calibration_seconds"] = row[
                "candidate_calibration_seconds"
            ]
            flattened["candidate_evaluation_seconds"] = row[
                "candidate_evaluation_seconds"
            ]
            writer.writerow(flattened)

    report_lines = [
        "# Phase 3 localization experiment",
        "",
        "- Candidate: nearest-neighbor upsampling of the existing 28x28 raw patch-distance map.",
        "- Baseline: bilinear upsampling of the same raw map.",
        "- No feature extractor, memory bank, patch distance, image score, or preprocessing change.",
        "- Candidate pixel thresholds use only normal official `train/good` maps at the existing 99.5th percentile. The image threshold remains the saved baseline normal-only threshold.",
        "- Official test data is used only for final evaluation and representative diagnostics.",
        "",
        "## Per-category metrics",
        "",
        "| Category | Image AUROC | Image AP | Image precision | Image recall | Image F1 | Pixel AUROC | Pixel AP | Pixel precision | Pixel recall | Pixel F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        before, after = row["baseline"], row["candidate"]
        report_lines.append(
            f"| {row['category']} "
            f"| {before['image_auroc']:.4f} → {after['image_auroc']:.4f} "
            f"| {before['image_average_precision']:.4f} → {after['image_average_precision']:.4f} "
            f"| {before['image_precision']:.4f} → {after['image_precision']:.4f} "
            f"| {before['image_recall']:.4f} → {after['image_recall']:.4f} "
            f"| {before['image_f1']:.4f} → {after['image_f1']:.4f} "
            f"| {before['pixel_auroc']:.4f} → {after['pixel_auroc']:.4f} "
            f"| {before['pixel_average_precision']:.4f} → {after['pixel_average_precision']:.4f} "
            f"| {before['pixel_precision']:.4f} → {after['pixel_precision']:.4f} "
            f"| {before['pixel_recall']:.4f} → {after['pixel_recall']:.4f} "
            f"| {before['pixel_f1']:.4f} → {after['pixel_f1']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Macro averages",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        before = baseline_macro.get(metric)
        after = candidate_macro.get(metric)
        delta = comparison["macro_average"]["delta_candidate_minus_baseline"][
            metric
        ]
        if before is not None and after is not None:
            report_lines.append(
                f"| {metric} | {before:.6f} | {after:.6f} | {delta:+.6f} |"
            )
    report_lines.extend(
        [
            "",
            "## Training, calibration, and evaluation time",
            "",
            "| Category | Baseline fit seconds | Candidate fit seconds | Baseline calibration seconds | Candidate calibration seconds | Baseline evaluation seconds | Candidate evaluation seconds |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        baseline_stages = row["baseline_stage_durations"] or {}
        report_lines.append(
            f"| {row['category']} "
            f"| {baseline_stages.get('training_seconds', 'not recorded')} "
            f"| not refit (baseline bank reused) "
            f"| {baseline_stages.get('calibration_seconds', 'not recorded')} "
            f"| {row['candidate_calibration_seconds']:.3f} "
            f"| {baseline_stages.get('evaluation_seconds', 'not recorded')} "
            f"| {row['candidate_evaluation_seconds']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## CPU latency and memory",
            "",
            "| Category | Configuration | Samples | Mean ms | Median ms | P95 ms | Throughput images/s | Peak working set bytes |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        for label, benchmark in (
            ("baseline", row["baseline_benchmark"]),
            ("nearest", row["candidate_benchmark"]),
        ):
            timing = benchmark["timings"]["end_to_end_prediction"]
            report_lines.append(
                f"| {row['category']} | {label} | {benchmark['image_count']} "
                f"| {timing['mean_ms']:.3f} | {timing['median_ms']:.3f} "
                f"| {timing['p95_ms']:.3f} | "
                f"{benchmark['throughput_images_per_second']:.3f} "
                f"| {benchmark['cpu_peak_working_set_bytes']} |"
            )
    report_lines.extend(
        [
            "",
            "## Runtime and limitations",
            "",
            "- Candidate fitting time is not applicable: the baseline memory bank and backbone were reused, not refit.",
            "- Per-category normal-only calibration and test evaluation durations are recorded in `comparison.json` and each category's `evaluation.json`.",
            "- Nearest interpolation changes only spatial rendering of each existing patch score; it cannot restore image information lost during 224x224 preprocessing or produce finer semantic features than the 28x28 grid.",
            "- Diagnostic images are stored under each category's `examples/true_positive`, `false_positive`, and `false_negative` directory when that case exists. A missing case means there was no such test prediction at the fixed threshold.",
            "",
        ]
    )
    (output_dir / "evaluation_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return comparison


def _run_category(
    category: str,
    dataset_root: Path,
    output_dir: Path,
    baseline_result: dict,
    tracking_uri: str,
    experiment_name: str,
    warmup: int,
    samples: int,
) -> dict:
    validate_dataset(dataset_root, category)
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {experiment_name}")
    run_names = {
        f"phase3-localization-nearest-{category}",
        f"phase3-localization-benchmark-bilinear-{category}",
        f"phase3-localization-benchmark-nearest-{category}",
    }
    existing_runs = client.search_runs([experiment.experiment_id], max_results=1000)
    existing_names = {run.data.tags.get("mlflow.runName") for run in existing_runs}
    duplicates = sorted(run_names & existing_names)
    if duplicates:
        raise RuntimeError(
            "Refusing to create duplicate MLflow runs: " + ", ".join(duplicates)
        )

    baseline_model_path = Path("artifacts") / category / "model.npz"
    if not baseline_model_path.is_file():
        raise FileNotFoundError(f"Baseline model not found: {baseline_model_path}")
    category_root = output_dir / category
    model_path = category_root / "model.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = PatchCoreModel.load(baseline_model_path, device="cpu")
    _save_localization_diagnostics(
        model,
        baseline_result,
        dataset_root,
        category,
        category_root / "examples" / "baseline",
    )
    model.config.dataset_root = str(dataset_root)
    model.config.category = category
    model.config.anomaly_map_upsampling = "nearest"
    model.config.model_path = str(model_path)
    if model.memory_bank is None or model.threshold is None:
        raise ValueError(f"Baseline model is not fitted and calibrated: {category}")

    train_paths = load_train_images(dataset_root, category)
    calibration_started = perf_counter()
    model.pixel_threshold = _determine_nearest_pixel_threshold(
        train_paths, model, quantile=0.995
    )
    calibration_seconds = perf_counter() - calibration_started
    model.save(model_path)

    evaluation_started = perf_counter()
    result = evaluate_category(
        model,
        dataset_root,
        category,
        threshold=model.threshold,
        pixel_threshold=model.pixel_threshold,
    )
    evaluation_seconds = perf_counter() - evaluation_started
    result.update(
        {
            "status": "measured",
            "experiment_id": EXPERIMENT_ID,
            "dataset_type": "VisA",
            "dataset_identity": str(dataset_root.resolve()),
            "evaluation_split": "official test",
            "baseline_model_artifact": str(baseline_model_path),
            "candidate_model_artifact": str(model_path),
            "feature_grid": [28, 28],
            "feature_dimensions": 1536,
            "anomaly_map_upsampling": "nearest",
            "fitting_strategy": (
                "reused the existing memory bank fitted from official train/good; "
                "no candidate refit"
            ),
            "calibration_split": "official train/good normal images only",
            "pixel_threshold_quantile": 0.995,
            "calibration_seconds": calibration_seconds,
            "evaluation_seconds": evaluation_seconds,
            "memory_bank_bytes": int(model.memory_bank.nbytes),
            "model_artifact_bytes": model_path.stat().st_size,
            "random_seed": model.random_seed,
        }
    )
    result["diagnostic_examples"] = _save_localization_diagnostics(
        model,
        result,
        dataset_root,
        category,
        category_root / "examples" / "nearest",
    )
    result_path = category_root / "evaluation.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    config_path = category_root / "experiment_config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "category": category,
                "dataset_identity": str(dataset_root.resolve()),
                "dataset_type": "VisA",
                "training_split": "official train/good; baseline fitted bank reused",
                "calibration_split": "official train/good normal images only",
                "evaluation_split": "official test",
                "random_seed": model.random_seed,
                "image_size": model.input_size,
                "feature_grid": [28, 28],
                "feature_dimensions": 1536,
                "baseline_upsampling": "bilinear",
                "candidate_upsampling": "nearest",
                "image_threshold": model.threshold,
                "pixel_threshold": model.pixel_threshold,
                "pixel_threshold_quantile": 0.995,
                "candidate_memory_bank": "unchanged fitted baseline bank",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = {
        key: float(value)
        for key, value in result.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }
    log_experiment(
        run_name=f"phase3-localization-nearest-{category}",
        params={
            "experiment_id": EXPERIMENT_ID,
            "category": category,
            "dataset_type": "VisA",
            "dataset_identity": str(dataset_root.resolve()),
            "model_version": "baseline-bank-nearest-map-upsampling",
            "training_split": "official train/good; baseline fitted bank reused",
            "calibration_split": "official train/good normal images only",
            "evaluation_split": "official test",
            "random_seed": model.random_seed,
            "image_size": model.input_size,
            "feature_grid": "28x28",
            "feature_dimensions": 1536,
            "baseline_upsampling": "bilinear",
            "candidate_upsampling": "nearest",
            "pixel_threshold_quantile": 0.995,
            "memory_bank_bytes": int(model.memory_bank.nbytes),
        },
        metrics=metrics,
        artifact_paths=(model_path, config_path, result_path, category_root / "examples"),
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    benchmark_results = {}
    for label, artifact_path in (
        ("bilinear", baseline_model_path),
        ("nearest", model_path),
    ):
        benchmark_path = category_root / "benchmark" / f"{label}.json"
        benchmark_results[label] = _benchmark(
            artifact_path,
            dataset_root,
            category,
            benchmark_path,
            warmup=warmup,
            samples=samples,
            run_name=f"phase3-localization-benchmark-{label}-{category}",
        )
        benchmark_results[label]["model_path"] = str(artifact_path)
        benchmark_results[label]["anomaly_map_upsampling"] = (
            "bilinear" if label == "bilinear" else "nearest"
        )
    result["benchmark"] = benchmark_results["nearest"]
    baseline_result["benchmark"] = benchmark_results["bilinear"]
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled VisA localization experiment comparing baseline "
            "bilinear and nearest anomaly-map upsampling."
        )
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "experiments" / EXPERIMENT_ID,
    )
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse categories with measured evaluation and both benchmark outputs.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")

    config = load_config_from_env()
    dataset_root = args.dataset_root or Path(config.dataset_root)
    tracking_uri = args.tracking_uri or config.mlflow_tracking_uri
    experiment_name = args.experiment_name or config.mlflow_experiment
    if not tracking_uri:
        parser.error("Set VISIONGUARD_MLFLOW_TRACKING_URI or pass --tracking-uri.")

    baseline_payload = json.loads(
        Path("reports/evaluation_results.json").read_text(encoding="utf-8")
    )
    if baseline_payload.get("status") != "measured":
        raise ValueError("The preserved baseline evaluation is not fully measured.")
    baseline_rows = {
        row["category"]: row for row in baseline_payload["categories"]
    }
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {experiment_name}")
    mlflow_runs = client.search_runs([experiment.experiment_id], max_results=1000)
    categories = list(dict.fromkeys(args.categories))
    candidates = {}
    failures = []
    for category in categories:
        print(f"Phase 3 category: {category}", flush=True)
        if category not in baseline_rows:
            failures.append(
                {"category": category, "error": "Category missing from baseline report"}
            )
            continue
        baseline_rows[category] = dict(baseline_rows[category])
        model_path_suffix = f"artifacts/{category}/model.npz"
        category_runs = [
            run
            for run in mlflow_runs
            if run.data.params.get("category") == category
            and run.data.params.get("dataset_identity")
            == str(dataset_root.resolve())
        ]
        stage_durations = {}
        for stage in ("train", "calibration", "evaluation"):
            stage_runs = [
                run
                for run in category_runs
                if run.data.tags.get("mlflow.runName") == f"{category}-{stage}"
            ]
            if stage == "train":
                stage_runs = [
                    run
                    for run in stage_runs
                    if run.data.params.get("model_path", "")
                    .replace("\\", "/")
                    .endswith(model_path_suffix)
                ]
            if stage_runs:
                latest = max(stage_runs, key=lambda run: run.info.start_time)
                metric_key = (
                    "training_seconds" if stage == "train" else f"{stage}_seconds"
                )
                stage_durations[metric_key] = latest.data.metrics.get(metric_key)
        baseline_rows[category]["phase3_stage_durations"] = stage_durations
        category_root = args.output_dir / category
        evaluation_path = category_root / "evaluation.json"
        baseline_benchmark_path = category_root / "benchmark" / "bilinear.json"
        candidate_benchmark_path = category_root / "benchmark" / "nearest.json"
        if (
            args.resume
            and evaluation_path.is_file()
            and baseline_benchmark_path.is_file()
            and candidate_benchmark_path.is_file()
        ):
            existing = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "measured"
                and existing.get("experiment_id") == EXPERIMENT_ID
                and existing.get("anomaly_map_upsampling") == "nearest"
            ):
                baseline_rows[category]["benchmark"] = json.loads(
                    baseline_benchmark_path.read_text(encoding="utf-8")
                )
                existing["benchmark"] = json.loads(
                    candidate_benchmark_path.read_text(encoding="utf-8")
                )
                candidates[category] = existing
                _write_comparison(
                    {name: baseline_rows[name] for name in candidates},
                    candidates,
                    args.output_dir,
                )
                print(f"Resumed measured category: {category}", flush=True)
                continue
        try:
            candidates[category] = _run_category(
                category,
                dataset_root,
                args.output_dir,
                baseline_rows[category],
                tracking_uri,
                experiment_name,
                args.warmup,
                args.samples,
            )
            print(f"Completed category: {category}", flush=True)
        except Exception as exc:
            failures.append(
                {"category": category, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"Failed category {category}: {type(exc).__name__}: {exc}", flush=True)
        if candidates:
            comparison = _write_comparison(
                {
                    name: baseline_rows[name]
                    for name in candidates
                },
                candidates,
                args.output_dir,
            )
            comparison["failed_categories"] = failures
            (args.output_dir / "comparison.json").write_text(
                json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
            )
    if failures or len(candidates) != len(categories):
        raise SystemExit(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "completed_categories": sorted(candidates),
                    "failed_categories": failures,
                    "output_dir": str(args.output_dir),
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "completed_categories": sorted(candidates),
                "comparison": str(args.output_dir / "comparison.json"),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
