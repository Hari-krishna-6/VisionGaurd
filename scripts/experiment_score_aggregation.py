from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from scripts.evaluate import _write_reports
from visionguard.config import load_config_from_env
from visionguard.data import load_test_items, load_train_images, validate_dataset
from visionguard.evaluation import determine_threshold, evaluate_category
from visionguard.mlflow_utils import log_experiment
from visionguard.model import PatchCoreModel
from visionguard.visualization import generate_heatmap, overlay_heatmap

EXPERIMENT_ID = "patchcore_topk_mean_1pct_v1"
DEFAULT_CATEGORIES = ("candle", "macaroni1", "macaroni2")


def _save_diagnostic_examples(
    model: PatchCoreModel,
    result: dict,
    dataset_root: Path,
    category: str,
    output_dir: Path,
) -> list[dict]:
    predictions = {row["image_path"]: row for row in result["image_predictions"]}
    items = load_test_items(dataset_root, category)
    item_by_path = {str(item.image_path): item for item in items}
    selected: dict[str, dict] = {}
    for prediction in predictions.values():
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

    examples = []
    for kind, prediction in selected.items():
        item = item_by_path[prediction["image_path"]]
        score, raw_map = model.predict_image(item.image_path)
        example_dir = output_dir / kind
        example_dir.mkdir(parents=True, exist_ok=True)
        original_path = example_dir / "original.png"
        heatmap_path = example_dir / "heatmap.png"
        overlay_path = example_dir / "overlay.png"
        raw_scores_path = example_dir / "raw_scores.npy"
        mask_path = example_dir / "ground_truth.png" if item.mask is not None else None
        item.image.save(original_path)
        Image.fromarray(generate_heatmap(raw_map, item.image.size)).save(heatmap_path)
        Image.fromarray(overlay_heatmap(item.image, raw_map)).save(overlay_path)
        np.save(raw_scores_path, raw_map)
        if mask_path is not None:
            Image.fromarray(item.mask).save(mask_path)
        example = {
            "kind": kind,
            "split": "test",
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
        (example_dir / "metadata.json").write_text(
            json.dumps(example, indent=2) + "\n", encoding="utf-8"
        )
        examples.append(example)
    return examples


def _run_category(
    category: str,
    dataset_root: Path,
    baseline_model_path: Path,
    output_dir: Path,
    tracking_uri: str,
    experiment_name: str,
) -> dict:
    validate_dataset(dataset_root, category)
    if not baseline_model_path.is_file():
        raise FileNotFoundError(
            f"Baseline model not found: {baseline_model_path}. "
            "This experiment reuses the fitted baseline memory bank."
        )

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {experiment_name}")
    run_name = f"phase2-{EXPERIMENT_ID}-{category}"
    existing = client.search_runs([experiment.experiment_id], max_results=1000)
    if any(run.data.tags.get("mlflow.runName") == run_name for run in existing):
        raise RuntimeError(
            f"MLflow run {run_name} already exists; refusing to create a duplicate."
        )

    category_output = output_dir / category
    candidate_model_path = (
        Path("artifacts")
        / "experiments"
        / EXPERIMENT_ID
        / category
        / "model.npz"
    )
    candidate_model_path.parent.mkdir(parents=True, exist_ok=True)
    model = PatchCoreModel.load(baseline_model_path, device="cpu")
    model.config.dataset_root = str(dataset_root)
    model.config.category = category
    model.config.image_score_aggregation = "topk_mean"
    model.config.image_score_top_k_fraction = 0.01
    model.config.model_path = str(candidate_model_path)
    model.threshold = None

    train_paths = load_train_images(dataset_root, category)
    calibration_started = perf_counter()
    model.threshold = determine_threshold(train_paths, model, quantile=0.99)
    calibration_seconds = perf_counter() - calibration_started
    model.save(candidate_model_path)

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
            "model_version": "baseline-fitted-memory-bank+topk-mean-1pct",
            "dataset_type": "VisA",
            "dataset_identity": str(dataset_root.resolve()),
            "evaluation_split": "test",
            "image_score_aggregation": "mean of highest 1% spatial patch distances",
            "threshold_calibration": {
                "image_quantile": 0.99,
                "image_data": "normal train/good images only",
                "pixel_threshold": model.pixel_threshold,
                "pixel_threshold_source": (
                    "reused baseline calibration on normal train/good maps; "
                    "candidate changes image aggregation only"
                ),
            },
            "candidate_model_artifact": str(candidate_model_path),
            "model_artifact_bytes": candidate_model_path.stat().st_size,
            "memory_bank_bytes": int(model.memory_bank.nbytes),
            "calibration_seconds": calibration_seconds,
            "evaluation_seconds": evaluation_seconds,
        }
    )
    result["diagnostic_examples"] = _save_diagnostic_examples(
        model, result, dataset_root, category, category_output / "examples"
    )
    category_output.mkdir(parents=True, exist_ok=True)
    result_path = category_output / "evaluation.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    numeric_metrics = {
        key: float(value)
        for key, value in result.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }
    log_experiment(
        run_name=run_name,
        params={
            "experiment_id": EXPERIMENT_ID,
            "model_version": result["model_version"],
            "category": category,
            "dataset_type": "VisA",
            "dataset_identity": str(dataset_root.resolve()),
            "training_split": "official train/good; baseline memory bank reused",
            "calibration_split": "official train/good normal images only",
            "evaluation_split": "official test",
            "image_score_aggregation": "topk_mean",
            "image_score_top_k_fraction": 0.01,
            "image_threshold_quantile": 0.99,
            "pixel_threshold_source": "baseline normal train/good calibration",
            "random_seed": model.random_seed,
            "image_size": model.input_size,
            "sample_ratio": model.sample_ratio,
            "baseline_model_artifact": str(baseline_model_path),
            "candidate_model_artifact": str(candidate_model_path),
            "fitting_strategy": "reuse baseline fitted memory bank; no retraining",
        },
        metrics=numeric_metrics,
        artifact_paths=(
            candidate_model_path,
            result_path,
            category_output / "examples",
        ),
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PatchCore top-1%-mean image scoring using existing VisA "
            "models; thresholds use normal training images only."
        )
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse measured candidate results already present for requested categories.",
    )
    args = parser.parse_args()

    config = load_config_from_env()
    dataset_root = args.dataset_root or Path(config.dataset_root)
    output_dir = args.output_dir or Path("reports") / "experiments" / EXPERIMENT_ID
    tracking_uri = args.tracking_uri or config.mlflow_tracking_uri
    experiment_name = args.experiment_name or config.mlflow_experiment
    if not tracking_uri:
        parser.error("Set VISIONGUARD_MLFLOW_TRACKING_URI or pass --tracking-uri.")

    results = []
    failures = []
    categories = list(dict.fromkeys(args.categories))
    for category in categories:
        existing_result_path = output_dir / category / "evaluation.json"
        if args.resume and existing_result_path.is_file():
            existing_result = json.loads(
                existing_result_path.read_text(encoding="utf-8")
            )
            if (
                existing_result.get("status") == "measured"
                and existing_result.get("experiment_id") == EXPERIMENT_ID
                and existing_result.get("dataset_identity")
                == str(dataset_root.resolve())
            ):
                results.append(existing_result)
                _write_reports(results, failures, output_dir, categories)
                continue
        baseline_path = Path("artifacts") / category / "model.npz"
        try:
            result = _run_category(
                category,
                dataset_root,
                baseline_path,
                output_dir,
                tracking_uri,
                experiment_name,
            )
            results.append(result)
        except Exception as exc:
            failures.append(
                {
                    "category": category,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        _write_reports(results, failures, output_dir, categories)
        report_path = output_dir / "evaluation_results.json"
        if report_path.is_file():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["experiment_id"] = EXPERIMENT_ID
            payload["model_version"] = (
                "baseline-fitted-memory-bank+topk-mean-1pct"
            )
            payload["selection_protocol"] = (
                "Configuration and top-1% aggregation fixed before looking "
                "at official test labels. Thresholds calibrated on normal "
                "train/good images only; test split used once for measurement."
            )
            report_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "results": results,
                "failed_categories": failures,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    if failures or not results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
