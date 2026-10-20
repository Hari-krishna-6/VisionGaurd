from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
from sklearn import metrics

from .data import load_test_items, load_train_images, validate_dataset
from .model import PatchCoreModel

IMAGE_METRIC_KEYS = (
    "image_auroc",
    "image_average_precision",
    "image_precision",
    "image_recall",
    "image_f1",
    "image_false_positive_rate",
    "image_false_negative_rate",
)
PIXEL_METRIC_KEYS = (
    "pixel_auroc",
    "pixel_average_precision",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_false_positive_rate",
    "pixel_false_negative_rate",
)


def determine_threshold(
    train_image_paths: Iterable[str | Path],
    model: PatchCoreModel,
    quantile: float = 0.99,
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    scores = [model.predict_image(path)[0] for path in train_image_paths]
    if not scores:
        raise ValueError("Cannot calibrate image threshold without normal training images")
    return float(np.quantile(scores, quantile))


def determine_pixel_threshold(
    train_image_paths: Iterable[str | Path],
    model: PatchCoreModel,
    quantile: float = 0.995,
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    scores = [model.predict_image(path)[1].reshape(-1) for path in train_image_paths]
    if not scores:
        raise ValueError("Cannot calibrate pixel threshold without normal training images")
    return float(np.quantile(np.concatenate(scores), quantile))


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights)
    if weights.ndim != 1 or weights.size != values.size:
        raise ValueError("Weights must be a 1D array matching values")
    if not np.issubdtype(weights.dtype, np.integer):
        raise ValueError("Weights must be integers")
    weights = weights.astype(np.int64, copy=False)
    if not np.isfinite(values).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Values must be finite and weights must have positive mass")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative_weights = np.cumsum(weights[order], dtype=np.int64)
    position = quantile * (int(cumulative_weights[-1]) - 1)

    def value_at_rank(rank: int) -> float:
        index = int(np.searchsorted(cumulative_weights, rank, side="right"))
        return float(sorted_values[index])

    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    fraction = position - lower_rank
    return value_at_rank(lower_rank) * (1 - fraction) + value_at_rank(
        upper_rank
    ) * fraction


def _binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    prefix: str,
) -> Dict[str, object]:
    predictions = scores >= threshold
    tn, fp, fn, tp = metrics.confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    result: Dict[str, object] = {
        f"{prefix}_precision": float(metrics.precision_score(labels, predictions, zero_division=0)),
        f"{prefix}_recall": float(metrics.recall_score(labels, predictions, zero_division=0)),
        f"{prefix}_f1": float(metrics.f1_score(labels, predictions, zero_division=0)),
        f"{prefix}_false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        f"{prefix}_false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        f"{prefix}_confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if np.unique(labels).size == 2:
        result[f"{prefix}_auroc"] = float(metrics.roc_auc_score(labels, scores))
        result[f"{prefix}_average_precision"] = float(metrics.average_precision_score(labels, scores))
        result[f"{prefix}_auroc_status"] = "computed"
    else:
        result[f"{prefix}_auroc"] = None
        result[f"{prefix}_average_precision"] = None
        result[f"{prefix}_auroc_status"] = "undefined: labels contain only one class"
    return result


def evaluate_category(
    model: PatchCoreModel,
    dataset_root: str | Path,
    category: str,
    threshold: Optional[float] = None,
    pixel_threshold: Optional[float] = None,
) -> Dict[str, object]:
    validate_dataset(dataset_root, category=category)
    test_items = load_test_items(dataset_root, category)
    if not test_items:
        raise ValueError(f"No test items found for '{category}'")

    train_paths = load_train_images(dataset_root, category)
    if threshold is None:
        threshold = determine_threshold(train_paths, model)
    if pixel_threshold is None:
        pixel_threshold = determine_pixel_threshold(train_paths, model)
    if not np.isfinite(threshold) or not np.isfinite(pixel_threshold):
        raise ValueError("Evaluation thresholds must be finite")

    image_labels = []
    image_scores = []
    inference_seconds = []
    pixel_targets = []
    pixel_scores = []
    missing_masks = []
    image_predictions = []
    for item in test_items:
        started = perf_counter()
        score, anomaly_map = model.predict_image(item.image_path)
        inference_seconds.append(perf_counter() - started)
        label = 1 if item.label != "good" else 0
        image_labels.append(label)
        image_scores.append(score)
        image_predictions.append(
            {
                "image_path": str(item.image_path),
                "label": label,
                "defect_category": item.label,
                "score": score,
                "threshold": float(threshold),
                "prediction": int(score >= threshold),
                "mask_path": str(item.mask_path) if item.mask_path else None,
            }
        )

        if item.label != "good" and item.mask_path is None:
            missing_masks.append(str(item.image_path))
            continue
        if item.label == "good":
            mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
        else:
            raw_mask = item.mask
            if raw_mask is None:
                missing_masks.append(str(item.image_path))
                continue
            if raw_mask.ndim == 3:
                raw_mask = np.any(raw_mask > 0, axis=-1)
            mask = cv2.resize(
                raw_mask.astype(np.uint8),
                (anomaly_map.shape[1], anomaly_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask > 0).astype(np.uint8)
        pixel_targets.append(mask.reshape(-1))
        pixel_scores.append(anomaly_map.reshape(-1))

    labels = np.asarray(image_labels, dtype=np.uint8)
    scores = np.asarray(image_scores, dtype=np.float64)
    image_result = _binary_metrics(labels, scores, threshold, "image")

    pixel_result: Dict[str, object]
    if missing_masks:
        pixel_result = {
            **{f"{name}": None for name in PIXEL_METRIC_KEYS},
            "pixel_confusion_matrix": None,
            "pixel_auroc_status": (
                f"not computed: {len(missing_masks)} anomalous image(s) lack masks"
            ),
        }
    elif not pixel_targets:
        pixel_result = {
            **{name: None for name in PIXEL_METRIC_KEYS},
            "pixel_confusion_matrix": None,
            "pixel_auroc_status": "not computed: no pixel ground truth",
        }
    else:
        flat_labels = np.concatenate(pixel_targets)
        flat_scores = np.concatenate(pixel_scores)
        pixel_result = _binary_metrics(
            flat_labels,
            flat_scores,
            float(pixel_threshold),
            "pixel",
        )
    return {
        "category": category,
        "image_threshold": float(threshold),
        "pixel_threshold": float(pixel_threshold),
        **image_result,
        "confusion_matrix": image_result["image_confusion_matrix"],
        **pixel_result,
        "missing_mask_count": len(missing_masks),
        "missing_mask_images": missing_masks,
        "test_image_count": len(test_items),
        "inference_seconds_per_image": float(np.mean(inference_seconds)),
        "inference_sample_count": len(inference_seconds),
        "image_predictions": image_predictions,
        "threshold_calibration": {
            "image_quantile": 0.99,
            "pixel_quantile": 0.995,
            "data": "normal train/good images only",
        },
    }


def evaluate_model(
    model: PatchCoreModel,
    dataset_root: str | Path,
    category: str,
    threshold: Optional[float] = None,
    pixel_threshold: Optional[float] = None,
) -> Dict[str, object]:
    return evaluate_category(
        model,
        dataset_root,
        category,
        threshold=threshold,
        pixel_threshold=pixel_threshold,
    )


def macro_average(results: Iterable[Dict[str, object]]) -> Dict[str, Optional[float]]:
    rows = list(results)
    summary: Dict[str, Optional[float]] = {}
    for key in (*IMAGE_METRIC_KEYS, *PIXEL_METRIC_KEYS):
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float))
        ]
        summary[key] = float(np.mean(values)) if values else None
    return summary
