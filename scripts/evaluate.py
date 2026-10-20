from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image

from visionguard.config import ModelConfig, load_config_from_env
from visionguard.data import (
    build_synthetic_mvtec_dataset,
    list_categories,
    load_test_items,
    validate_dataset,
)
from visionguard.evaluation import evaluate_category, macro_average
from visionguard.mlflow_utils import log_experiment
from visionguard.model import PatchCoreModel
from visionguard.visualization import generate_heatmap, overlay_heatmap

METRIC_COLUMNS = (
    "image_auroc",
    "image_average_precision",
    "image_precision",
    "image_recall",
    "image_f1",
    "image_false_positive_rate",
    "image_false_negative_rate",
    "pixel_auroc",
    "pixel_average_precision",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_false_positive_rate",
    "pixel_false_negative_rate",
)


def _dataset_type(dataset_root: Path) -> str:
    return (
        "VisA"
        if "visa" in str(dataset_root).replace("\\", "/").lower()
        else "MVTec AD"
    )


def _evaluate(
    config: ModelConfig,
    dataset_root: Path,
    category: str,
    model_path: Path,
    dataset_type: str,
    output_dir: Path,
) -> dict:
    validate_dataset(dataset_root, category)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'. Run training first."
        )

    model = PatchCoreModel.load(model_path, device=config.device)
    started = perf_counter()
    result = evaluate_category(
        model,
        dataset_root,
        category,
        threshold=config.threshold if config.threshold is not None else model.threshold,
        pixel_threshold=model.pixel_threshold,
    )
    evaluation_seconds = perf_counter() - started
    result.update(
        {
            "status": "measured",
            "dataset_type": dataset_type,
            "dataset_identity": str(dataset_root.resolve()),
            "evaluation_split": "test",
            "evaluation_seconds": evaluation_seconds,
            "model_artifact": str(model_path),
        }
    )
    result["representative_examples"] = _save_representative_examples(
        model, dataset_root, category, output_dir / "examples" / category
    )
    metrics = {
        key: value
        for key, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    with TemporaryDirectory(prefix="visionguard-evaluation-") as temporary_root:
        report_path = Path(temporary_root) / "evaluation.json"
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log_experiment(
            run_name=f"{category}-evaluation",
            params={
                **config.to_dict(),
                "sample_ratio": model.sample_ratio,
                "random_seed": model.random_seed,
                "image_size": model.input_size,
                "category": category,
                "dataset_type": dataset_type,
                "dataset_identity": str(dataset_root.resolve()),
                "training_split": "train/good",
                "evaluation_split": "test",
                "image_threshold_source": (
                    "configured threshold"
                    if config.threshold is not None
                    else "saved normal-training calibration"
                ),
                "pixel_threshold_source": "saved normal-training calibration",
            },
            metrics=metrics,
            artifact_paths=(model_path, report_path),
            tracking_uri=config.mlflow_tracking_uri,
            experiment_name=config.mlflow_experiment,
        )
    return result


def _save_representative_examples(
    model: PatchCoreModel,
    dataset_root: Path,
    category: str,
    output_dir: Path,
) -> list[dict]:
    items = load_test_items(dataset_root, category)
    selected = [
        next((item for item in items if item.label == "good"), None),
        next((item for item in items if item.label != "good"), None),
    ]
    examples = []
    for item in selected:
        if item is None:
            continue
        kind = "normal" if item.label == "good" else "defective"
        image = item.image
        score, raw_map = model.predict_image(item.image_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        original_path = output_dir / f"{kind}_original.png"
        heatmap_path = output_dir / f"{kind}_heatmap.png"
        overlay_path = output_dir / f"{kind}_overlay.png"
        raw_map_path = output_dir / f"{kind}_raw_scores.npy"
        image.save(original_path)
        Image.fromarray(
            generate_heatmap(raw_map, (image.height, image.width))
        ).save(heatmap_path)
        Image.fromarray(overlay_heatmap(image, raw_map)).save(overlay_path)
        np.save(raw_map_path, raw_map)
        mask_path = None
        if item.mask is not None:
            mask_path = output_dir / f"{kind}_ground_truth.png"
            Image.fromarray(item.mask).save(mask_path)
        examples.append(
            {
                "image_path": str(item.image_path),
                "label": item.label,
                "image_score": score,
                "original": str(original_path),
                "ground_truth_mask": str(mask_path) if mask_path else None,
                "raw_anomaly_scores": str(raw_map_path),
                "heatmap": str(heatmap_path),
                "overlay": str(overlay_path),
            }
        )
    return examples


def _write_reports(
    results: list[dict],
    failures: list[dict],
    output_dir: Path,
    requested_categories: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_categories = {result["category"] for result in results}
    failed_categories = {failure["category"] for failure in failures}
    pending_categories = [
        category
        for category in requested_categories or []
        if category not in completed_categories and category not in failed_categories
    ]
    payload = {
        "status": (
            "measured"
            if results and not failures and not pending_categories
            else "partial"
            if results
            else "failed"
        ),
        "dataset_type": results[0]["dataset_type"] if results else None,
        "evaluation_split": "test",
        "threshold_selection": "99th/99.5th percentile of train/good normal scores",
        "categories": results,
        "macro_average": macro_average(results),
        "failed_categories": failures,
        "pending_categories": pending_categories,
    }
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "image_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        columns = (
            "category",
            "image_path",
            "label",
            "defect_category",
            "score",
            "threshold",
            "prediction",
            "mask_path",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            for prediction in result["image_predictions"]:
                writer.writerow({"category": result["category"], **prediction})
    with (output_dir / "per_category_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        columns = ("category", "status", "image_threshold", "pixel_threshold", *METRIC_COLUMNS,
                   "test_image_count", "missing_mask_count")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in columns})
        for failure in failures:
            writer.writerow(
                {"category": failure["category"], "status": "failed"}
            )

    lines = [
        "# Evaluation report",
        "",
        f"- Status: `{payload['status']}`",
        f"- Dataset type: `{payload['dataset_type'] or 'none'}`",
        "- Fitting/calibration split: `train/good` only",
        "- Final evaluation split: `test`",
        f"- Thresholds: {payload['threshold_selection']}",
        "- Image detection and pixel localization are reported separately.",
        "- Pixel metrics are null when any anomalous test image lacks its ground-truth mask.",
        "",
        "| Category | Image AUROC | Image AP | Image F1 | Pixel AUROC | Pixel AP | Pixel F1 | Test images | Missing masks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        values = [result.get(key) for key in METRIC_COLUMNS]
        lines.append(
            f"| {result['category']} | {values[0]} | {values[1]} | {values[4]} | "
            f"{values[7]} | {values[8]} | {values[11]} | "
            f"{result['test_image_count']} | {result['missing_mask_count']} |"
        )
    for failure in failures:
        lines.append(f"| {failure['category']} | FAILED: {failure['error']} | | | | | | | |")
    lines += [
        "",
        "## Macro-average",
        "",
        "Equal-weight arithmetic mean across categories with a defined metric:",
        "",
        "```json",
        json.dumps(payload["macro_average"], indent=2),
        "```",
        "",
        "## Failed categories",
        "",
        json.dumps(failures, indent=2) if failures else "None.",
        "",
        "## Pending categories",
        "",
        ", ".join(pending_categories) if pending_categories else "None.",
        "",
    ]
    (output_dir / "evaluation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _evaluate_categories(
    config: ModelConfig,
    dataset_root: Path,
    categories: list[str],
    model_path: Path,
    dataset_type: str,
    output_dir: Path,
    resume: bool = False,
) -> int:
    results = []
    failures = []
    report_path = output_dir / "evaluation_results.json"
    if resume and report_path.is_file():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            prior.get("dataset_type") == dataset_type
            and prior.get("evaluation_split") == "test"
        ):
            requested = set(categories)
            results = [
                result
                for result in prior.get("categories", [])
                if result.get("category") in requested
                and result.get("status") == "measured"
                and result.get("dataset_type") == dataset_type
                and result.get("dataset_identity") == str(dataset_root.resolve())
            ]
            completed = {result["category"] for result in results}
            failures = [
                failure
                for failure in prior.get("failed_categories", [])
                if failure.get("category") in requested and failure.get("category") not in completed
            ]
    for category in categories:
        if any(result["category"] == category for result in results):
            continue
        category_model_path = (
            model_path.parent / category / model_path.name
            if len(categories) > 1
            or (model_path.parent / category / model_path.name).is_file()
            else model_path
        )
        category_config = replace(config, category=category)
        try:
            result = _evaluate(
                category_config,
                dataset_root,
                category,
                category_model_path,
                dataset_type,
                output_dir,
            )
        except Exception as exc:
            failures = [failure for failure in failures if failure["category"] != category]
            failures.append(
                {
                    "category": category,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_reports(results, failures, output_dir, categories)
            continue
        failures = [failure for failure in failures if failure["category"] != category]
        results.append(result)
        _write_reports(results, failures, output_dir, categories)
    _write_reports(results, failures, output_dir, categories)
    print(json.dumps({
        "results": results,
        "failed_categories": failures,
        "report_directory": str(output_dir),
    }, indent=2))
    return 1 if failures or not results else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained category models on MVTec AD test splits."
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--category")
    parser.add_argument("--all-categories", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible measured results already in the output report.",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a temporary synthetic dataset for code-path validation only.",
    )
    args = parser.parse_args()
    if args.all_categories and args.category:
        parser.error("--category and --all-categories cannot be combined")

    config = load_config_from_env()
    dataset_root = args.dataset_root or Path(config.dataset_root)
    model_path = args.model or Path(config.model_path)
    if args.smoke_test:
        with TemporaryDirectory(prefix="visionguard-smoke-") as temporary_root:
            smoke_root = Path(temporary_root) / "dataset"
            build_synthetic_mvtec_dataset(smoke_root, category=config.category)
            code = _evaluate_categories(
                config,
                smoke_root,
                [config.category],
                model_path,
                "synthetic smoke test (not MVTec AD)",
                args.output_dir,
                resume=False,
            )
    else:
        categories = (
            list_categories(dataset_root)
            if args.all_categories
            else [args.category or config.category]
        )
        code = _evaluate_categories(
            config,
            dataset_root,
            categories,
            model_path,
            _dataset_type(dataset_root),
            args.output_dir,
            resume=args.resume,
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
