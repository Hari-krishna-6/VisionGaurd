from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

from visionguard.config import ModelConfig, load_config_from_env
from visionguard.data import load_test_items, load_train_images, validate_dataset
from visionguard.evaluation import _binary_metrics, macro_average
from visionguard.experimental_multiscale import (
    MultiscalePatchCoreModel,
    combine_multiscale_maps,
)
from visionguard.features import preprocess_image, resize_anomaly_map
from visionguard.mlflow_utils import log_experiment
from visionguard.model import PatchCoreModel
from visionguard.visualization import generate_heatmap, overlay_heatmap

EXPERIMENT_ID = "phase4_multiscale_v1"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "reports" / "experiments" / EXPERIMENT_ID
CATEGORIES = ("candle", "macaroni1", "macaroni2")
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
HISTOGRAM_BINS = 262_144


def _dataset_path(config: ModelConfig) -> Path:
    path = Path(config.dataset_root)
    return path if path.is_absolute() else ROOT / path


def _runtime_config() -> ModelConfig:
    config = load_config_from_env()
    snapshot_path = ROOT / "reports" / "environment.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not _dataset_path(config).is_dir() and snapshot.get("dataset_root"):
            config.dataset_root = snapshot["dataset_root"]
        if not config.mlflow_tracking_uri and snapshot.get("mlflow_tracking_uri"):
            config.mlflow_tracking_uri = snapshot["mlflow_tracking_uri"]
    config.mlflow_experiment = "visionguard-visa"
    return config


def _config_for_category(category: str, baseline_path: Path, candidate_path: Path) -> ModelConfig:
    with np.load(baseline_path, allow_pickle=False) as artifact:
        baseline_metadata = json.loads(str(artifact["metadata"].item()))
    values = dict(baseline_metadata["config"])
    values.update(
        {
            "category": category,
            "device": "cpu",
            "dataset_root": str(_dataset_path(_runtime_config())),
            "model_path": str(candidate_path),
            "threshold": None,
            "anomaly_map_upsampling": "bilinear",
            "image_score_aggregation": "max",
            "random_seed": int(baseline_metadata["random_seed"]),
            "sample_ratio": float(baseline_metadata["sample_ratio"]),
            "image_size": int(baseline_metadata["input_size"]),
        }
    )
    return ModelConfig(**values)


def _log_once(
    run_name: str,
    params: dict,
    metrics: dict,
    artifacts: tuple[Path, ...],
    tracking_uri: str | None,
) -> bool:
    if not tracking_uri:
        return False
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise RuntimeError("MLflow is required to record this experiment")
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("visionguard-visa")
    if experiment is not None:
        existing = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1,
        )
        if existing:
            print(f"MLflow run already exists; not duplicating: {run_name}", flush=True)
            return False
    log_experiment(
        run_name=run_name,
        params=params,
        metrics=metrics,
        artifact_paths=artifacts,
        tracking_uri=tracking_uri,
        experiment_name="visionguard-visa",
    )
    return True


def _normal_calibration(
    model: MultiscalePatchCoreModel,
    train_paths: list[Path],
    image_quantile: float = 0.99,
    pixel_quantile: float = 0.995,
) -> tuple[float, float, dict]:
    if not train_paths:
        raise ValueError("Cannot calibrate without normal train/good images")
    patch_maps: list[tuple[np.ndarray, tuple[int, int]]] = []
    image_scores: list[float] = []
    map_min = math.inf
    map_max = -math.inf
    pixel_count = 0

    for path in train_paths:
        image_score, patch_map, source_shape = model.predict_patch_map(path)
        image_scores.append(image_score)
        patch_maps.append((patch_map, source_shape))
        resized = resize_anomaly_map(patch_map, source_shape, interpolation="bilinear")
        map_min = min(map_min, float(resized.min()))
        map_max = max(map_max, float(resized.max()))
        pixel_count += int(resized.size)

    image_threshold = float(np.quantile(image_scores, image_quantile))
    if not math.isfinite(map_min) or not math.isfinite(map_max):
        raise ValueError("Normal calibration maps contain no finite scores")
    if map_max <= map_min:
        pixel_threshold = map_min
        histogram_counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
        histogram_counts[0] = pixel_count
        bin_width = 0.0
    else:
        histogram_counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
        bin_width = (map_max - map_min) / HISTOGRAM_BINS
        for patch_map, source_shape in patch_maps:
            resized = resize_anomaly_map(
                patch_map, source_shape, interpolation="bilinear"
            )
            counts, _ = np.histogram(
                resized,
                bins=HISTOGRAM_BINS,
                range=(map_min, map_max),
            )
            histogram_counts += counts
        rank = pixel_quantile * (pixel_count - 1)
        bin_index = int(np.searchsorted(np.cumsum(histogram_counts), rank, side="right"))
        bin_index = min(bin_index, HISTOGRAM_BINS - 1)
        pixel_threshold = float(map_min + (bin_index + 0.5) * bin_width)

    details = {
        "calibration_data": "official normal train/good only",
        "normal_image_count": len(train_paths),
        "image_quantile": image_quantile,
        "pixel_quantile": pixel_quantile,
        "image_threshold_method": "exact NumPy linear quantile over normal image max scores",
        "pixel_threshold_method": (
            f"streamed {HISTOGRAM_BINS}-bin histogram over bilinearly upsampled "
            "normal training maps; threshold is bin midpoint"
        ),
        "pixel_threshold_max_bin_width": float(bin_width),
        "pixel_count": pixel_count,
        "normal_map_min": map_min,
        "normal_map_max": map_max,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
    }
    return image_threshold, pixel_threshold, details


def _align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    values = np.asarray(mask)
    if values.ndim == 3:
        values = np.any(values > 0, axis=-1).astype(np.uint8)
    if values.shape != shape:
        values = cv2.resize(
            values.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return (values > 0).astype(np.uint8) * 255


def _sorted_score_groups(
    sorted_keys: np.ndarray,
    sorted_labels: np.ndarray,
    total: int,
    chunk_size: int,
    descending: bool,
):
    cursor = total if descending else 0
    carry_key: int | None = None
    carry_positive = 0
    carry_total = 0
    while (cursor > 0 if descending else cursor < total):
        if descending:
            start = max(0, cursor - chunk_size)
            block_keys = np.asarray(sorted_keys[start:cursor])[::-1]
            block_labels = np.asarray(sorted_labels[start:cursor])[::-1]
            cursor = start
        else:
            stop = min(total, cursor + chunk_size)
            block_keys = np.asarray(sorted_keys[cursor:stop])
            block_labels = np.asarray(sorted_labels[cursor:stop])
            cursor = stop
        starts = np.r_[0, np.flatnonzero(block_keys[1:] != block_keys[:-1]) + 1]
        group_keys = block_keys[starts]
        group_totals = np.diff(np.r_[starts, block_keys.size]).astype(np.int64)
        group_positives = np.add.reduceat(
            block_labels.astype(np.uint64, copy=False), starts
        ).astype(np.int64)

        if carry_key is not None:
            if group_keys.size and int(group_keys[0]) == carry_key:
                group_totals[0] += carry_total
                group_positives[0] += carry_positive
            else:
                yield carry_key, carry_positive, carry_total
            carry_key = None
            carry_positive = 0
            carry_total = 0

        if group_keys.size > 1:
            for index in range(group_keys.size - 1):
                yield (
                    int(group_keys[index]),
                    int(group_positives[index]),
                    int(group_totals[index]),
                )
        if group_keys.size:
            last = group_keys.size - 1
            carry_key = int(group_keys[last])
            carry_positive = int(group_positives[last])
            carry_total = int(group_totals[last])
    if carry_key is not None:
        yield carry_key, carry_positive, carry_total


def _radix_sort_scores(
    score_values: np.memmap,
    labels: np.memmap,
    temporary_root: Path,
    chunk_size: int = 5_000_000,
) -> tuple[np.memmap, np.memmap]:
    total = score_values.size
    source_keys = score_values.view(np.uint32)
    source_labels = labels
    if np.any(score_values < 0):
        raise ValueError("Anomaly scores must be non-negative for exact radix ordering")

    first_keys = np.memmap(
        temporary_root / "radix-a-keys.bin",
        mode="w+",
        dtype=np.uint32,
        shape=(total,),
    )
    first_labels = np.memmap(
        temporary_root / "radix-a-labels.bin",
        mode="w+",
        dtype=np.uint8,
        shape=(total,),
    )
    second_keys = np.memmap(
        temporary_root / "radix-b-keys.bin",
        mode="w+",
        dtype=np.uint32,
        shape=(total,),
    )
    second_labels = np.memmap(
        temporary_root / "radix-b-labels.bin",
        mode="w+",
        dtype=np.uint8,
        shape=(total,),
    )

    for shift in (0, 8, 16, 24):
        counts = np.zeros(256, dtype=np.int64)
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)
            digits = ((source_keys[start:stop] >> shift) & 0xFF).astype(
                np.uint8, copy=False
            )
            counts += np.bincount(digits, minlength=256)
        cursors = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
        destination_keys, destination_labels = (
            (first_keys, first_labels)
            if source_keys is not first_keys
            else (second_keys, second_labels)
        )
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)
            digits = ((source_keys[start:stop] >> shift) & 0xFF).astype(
                np.uint8, copy=False
            )
            order = np.argsort(digits, kind="stable")
            ordered_digits = digits[order]
            ordered_keys = source_keys[start:stop][order]
            ordered_labels = source_labels[start:stop][order]
            boundaries = np.r_[
                0,
                np.flatnonzero(ordered_digits[1:] != ordered_digits[:-1]) + 1,
                ordered_digits.size,
            ]
            for index in range(boundaries.size - 1):
                left = int(boundaries[index])
                right = int(boundaries[index + 1])
                digit = int(ordered_digits[left])
                output_start = int(cursors[digit])
                output_stop = output_start + right - left
                destination_keys[output_start:output_stop] = ordered_keys[left:right]
                destination_labels[output_start:output_stop] = ordered_labels[left:right]
                cursors[digit] = output_stop
        destination_keys.flush()
        destination_labels.flush()
        source_keys = destination_keys
        source_labels = destination_labels

    return source_keys, source_labels


def _exact_pixel_ranking_metrics(
    sorted_keys: np.ndarray,
    sorted_labels: np.ndarray,
    total: int,
    chunk_size: int = 5_000_000,
) -> tuple[float | None, float | None, str]:
    positives = 0
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        positives += int(np.asarray(sorted_labels[start:stop]).sum(dtype=np.uint64))
    negatives = total - positives
    if not positives or not negatives:
        return None, None, "undefined: labels contain only one class"

    concordant_pairs = 0.0
    negatives_below = 0
    for _, positive_count, group_total in _sorted_score_groups(
        sorted_keys, sorted_labels, total, chunk_size, descending=False
    ):
        negative_count = group_total - positive_count
        concordant_pairs += positive_count * (
            negatives_below + negative_count / 2
        )
        negatives_below += negative_count
    auroc = concordant_pairs / (positives * negatives)

    true_positives = 0
    false_positives = 0
    average_precision = 0.0
    for _, positive_count, group_total in _sorted_score_groups(
        sorted_keys, sorted_labels, total, chunk_size, descending=True
    ):
        true_positives += positive_count
        false_positives += group_total - positive_count
        average_precision += (
            positive_count
            / positives
            * true_positives
            / (true_positives + false_positives)
        )
    return float(auroc), float(average_precision), "computed exactly by bounded-memory radix sort"


def _evaluate_category_bounded(
    model: MultiscalePatchCoreModel,
    dataset_root: Path,
    category: str,
    threshold: float,
    pixel_threshold: float,
) -> dict:
    validate_dataset(dataset_root, category)
    test_items = load_test_items(dataset_root, category)
    if not test_items:
        raise ValueError(f"No test items found for '{category}'")
    missing_masks = [
        str(item.image_path)
        for item in test_items
        if item.label != "good" and item.mask_path is None
    ]

    pixel_count = 0
    if not missing_masks:
        for item in test_items:
            with Image.open(item.image_path) as image:
                pixel_count += image.width * image.height

    image_labels: list[int] = []
    image_scores: list[float] = []
    image_predictions: list[dict] = []
    inference_seconds: list[float] = []
    temporary = tempfile.TemporaryDirectory(prefix="visionguard-pixel-rank-")
    pixel_scores = None
    pixel_labels = None
    sorted_keys = None
    sorted_labels = None
    try:
        temporary_root = Path(temporary.name)
        pixel_scores = (
            np.memmap(
                temporary_root / "scores.bin",
                mode="w+",
                dtype=np.float32,
                shape=(pixel_count,),
            )
            if pixel_count
            else None
        )
        pixel_labels = (
            np.memmap(
                temporary_root / "labels.bin",
                mode="w+",
                dtype=np.uint8,
                shape=(pixel_count,),
            )
            if pixel_count
            else None
        )
        offset = 0
        tn = fp = fn = tp = 0
        for item in test_items:
            started = perf_counter()
            score, anomaly_map = model.predict_image(item.image_path)
            inference_seconds.append(perf_counter() - started)
            label = int(item.label != "good")
            image_labels.append(label)
            image_scores.append(score)
            prediction = int(score >= threshold)
            image_predictions.append(
                {
                    "image_path": str(item.image_path),
                    "label": label,
                    "defect_category": item.label,
                    "score": score,
                    "threshold": threshold,
                    "prediction": prediction,
                    "mask_path": str(item.mask_path) if item.mask_path else None,
                }
            )

            if pixel_count == 0:
                continue
            if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
                raise ValueError(f"Invalid anomaly map for {item.image_path}")
            if item.label == "good":
                mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
            else:
                if item.mask is None:
                    raise ValueError(f"Missing ground-truth mask for {item.image_path}")
                mask = _align_mask(item.mask, anomaly_map.shape)
            if mask.shape != anomaly_map.shape:
                raise ValueError(f"Mask is not aligned for {item.image_path}")
            flattened_scores = anomaly_map.reshape(-1)
            flattened_labels = (mask.reshape(-1) > 0).astype(np.uint8)
            end = offset + flattened_scores.size
            if end > pixel_count:
                raise ValueError("Observed more test pixels than precomputed")
            pixel_scores[offset:end] = flattened_scores
            pixel_labels[offset:end] = flattened_labels
            predicted = flattened_scores >= pixel_threshold
            actual = flattened_labels.astype(bool, copy=False)
            tp += int(np.count_nonzero(predicted & actual))
            fp += int(np.count_nonzero(predicted & ~actual))
            fn += int(np.count_nonzero(~predicted & actual))
            tn += int(np.count_nonzero(~predicted & ~actual))
            offset = end

        if pixel_count and offset != pixel_count:
            raise ValueError(
                f"Wrote {offset} pixel scores but expected {pixel_count}"
            )
        image_metrics = _binary_metrics(
            np.asarray(image_labels, dtype=np.uint8),
            np.asarray(image_scores, dtype=np.float64),
            threshold,
            "image",
        )
        if missing_masks:
            pixel_metrics = {
                "pixel_auroc": None,
                "pixel_average_precision": None,
                "pixel_precision": None,
                "pixel_recall": None,
                "pixel_f1": None,
                "pixel_false_positive_rate": None,
                "pixel_false_negative_rate": None,
                "pixel_confusion_matrix": None,
                "pixel_auroc_status": (
                    f"not computed: {len(missing_masks)} anomalous image(s) lack masks"
                ),
            }
        else:
            pixel_scores.flush()
            pixel_labels.flush()
            sorted_keys, sorted_labels = _radix_sort_scores(
                pixel_scores,
                pixel_labels,
                temporary_root,
            )
            pixel_auroc, pixel_ap, ranking_status = _exact_pixel_ranking_metrics(
                sorted_keys,
                sorted_labels,
                pixel_count,
            )
            pixel_precision = tp / (tp + fp) if tp + fp else 0.0
            pixel_recall = tp / (tp + fn) if tp + fn else 0.0
            pixel_f1 = (
                2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall)
                if pixel_precision + pixel_recall
                else 0.0
            )
            pixel_metrics = {
                "pixel_auroc": pixel_auroc,
                "pixel_average_precision": pixel_ap,
                "pixel_precision": pixel_precision,
                "pixel_recall": pixel_recall,
                "pixel_f1": pixel_f1,
                "pixel_false_positive_rate": (
                    fp / (fp + tn) if fp + tn else None
                ),
                "pixel_false_negative_rate": (
                    fn / (fn + tp) if fn + tp else None
                ),
                "pixel_confusion_matrix": [
                    [int(tn), int(fp)],
                    [int(fn), int(tp)],
                ],
                "pixel_auroc_status": ranking_status,
            }
        return {
            "category": category,
            "image_threshold": float(threshold),
            "pixel_threshold": float(pixel_threshold),
            **image_metrics,
            "confusion_matrix": image_metrics["image_confusion_matrix"],
            **pixel_metrics,
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
            "pixel_metric_implementation": (
                "exact existing pixel metric definitions; full-resolution "
                "scores and labels external-memory radix sorted"
            ),
            "pixel_count": pixel_count,
        }
    finally:
        for memory_map in (
            sorted_keys,
            sorted_labels,
            pixel_scores,
            pixel_labels,
        ):
            if isinstance(memory_map, np.memmap):
                memory_map.flush()
                memory_map._mmap.close()
        temporary.cleanup()


def _save_candidate_diagnostics(
    model: MultiscalePatchCoreModel,
    result: dict,
    dataset_root: Path,
    category: str,
    output_dir: Path,
) -> dict[str, str | None]:
    items = {item.image_path.resolve(): item for item in load_test_items(dataset_root, category)}
    case_for_prediction = {
        (1, 1): "true_positive",
        (1, 0): "false_negative",
        (0, 1): "false_positive",
    }
    selected: dict[str, dict] = {}
    for prediction in result["image_predictions"]:
        case = case_for_prediction.get(
            (int(prediction["label"]), int(prediction["prediction"]))
        )
        if case is not None and case not in selected:
            selected[case] = prediction

    output_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str | None] = {}
    for case in ("true_positive", "false_negative", "false_positive"):
        prediction = selected.get(case)
        if prediction is None:
            index[case] = None
            continue
        item = items[Path(prediction["image_path"]).resolve()]
        image = item.image
        score, raw_map = model.predict_image(item.image_path)
        if raw_map.shape != (image.height, image.width):
            raise ValueError("Candidate anomaly map is not aligned to the source image")
        if item.label == "good":
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            if item.mask is None:
                raise ValueError(f"Missing official test mask for {item.image_path}")
            mask = _align_mask(item.mask, (image.height, image.width))
        case_dir = output_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        image.save(case_dir / "original.png")
        Image.fromarray(mask).save(case_dir / "ground_truth.png")
        Image.fromarray(
            generate_heatmap(raw_map, (image.height, image.width))
        ).save(case_dir / "heatmap.png")
        Image.fromarray(overlay_heatmap(image, raw_map)).save(case_dir / "overlay.png")
        metadata = {
            **prediction,
            "image_score_recomputed": score,
            "pixel_threshold": result["pixel_threshold"],
            "source_dimensions": {"width": image.width, "height": image.height},
            "mask_dimensions": {"width": image.width, "height": image.height},
            "anomaly_map_dimensions": {
                "width": int(raw_map.shape[1]),
                "height": int(raw_map.shape[0]),
            },
            "case": case,
            "split": "official test",
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        index[case] = str(case_dir.relative_to(OUTPUT_ROOT))
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return index


def _copy_baseline_diagnostics(category: str, target: Path) -> None:
    source = (
        ROOT
        / "reports"
        / "experiments"
        / "phase3_localization_v1"
        / category
        / "examples"
        / "baseline"
    )
    for case in ("true_positive", "false_negative", "false_positive"):
        source_case = source / case
        if not source_case.is_dir():
            continue
        target_case = target / case
        target_case.mkdir(parents=True, exist_ok=True)
        for filename in (
            "original.png",
            "ground_truth.png",
            "overlay.png",
            "metadata.json",
        ):
            source_file = source_case / filename
            if source_file.is_file():
                shutil.copy2(source_file, target_case / filename)
        raw_score_path = source_case / "raw_scores.npy"
        if not raw_score_path.is_file():
            raise FileNotFoundError(
                f"Baseline raw anomaly map is missing: {raw_score_path}"
            )
        raw_map = np.load(raw_score_path, mmap_mode="r", allow_pickle=False)
        try:
            with Image.open(target_case / "original.png") as source_image:
                image = source_image.convert("RGB")
            if raw_map.ndim != 2 or not np.isfinite(raw_map).all():
                raise ValueError(f"Invalid baseline anomaly map: {raw_score_path}")
            heatmap = generate_heatmap(
                raw_map, (image.height, image.width)
            )
            Image.fromarray(heatmap).save(target_case / "heatmap.png")
            Image.fromarray(overlay_heatmap(image, raw_map)).save(
                target_case / "overlay.png"
            )
            metadata_path = target_case / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "original": str(
                        (target_case / "original.png").relative_to(ROOT)
                    ),
                    "ground_truth_mask": str(
                        (target_case / "ground_truth.png").relative_to(ROOT)
                    ),
                    "heatmap": str(
                        (target_case / "heatmap.png").relative_to(ROOT)
                    ),
                    "overlay": str(
                        (target_case / "overlay.png").relative_to(ROOT)
                    ),
                    "raw_anomaly_scores": str(
                        raw_score_path.relative_to(ROOT)
                    ),
                    "source_dimensions": {
                        "width": image.width,
                        "height": image.height,
                    },
                    "anomaly_map_dimensions": {
                        "width": int(raw_map.shape[1]),
                        "height": int(raw_map.shape[0]),
                    },
                }
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            raw_map._mmap.close()


def _baseline_data(category: str) -> tuple[dict, dict, dict]:
    phase3_root = (
        ROOT / "reports" / "experiments" / "phase3_localization_v1" / category
    )
    phase3_comparison = json.loads(
        (
            ROOT
            / "reports"
            / "experiments"
            / "phase3_localization_v1"
            / "comparison.json"
        ).read_text(encoding="utf-8")
    )
    comparison_row = next(
        row for row in phase3_comparison["categories"] if row["category"] == category
    )
    phase3_evaluation = json.loads(
        (phase3_root / "evaluation.json").read_text(encoding="utf-8")
    )
    baseline_result = dict(comparison_row["baseline"])
    baseline_result.update(
        {
            "image_threshold": comparison_row["baseline_threshold"],
            "pixel_threshold": comparison_row["baseline_pixel_threshold"],
            "test_image_count": phase3_evaluation["test_image_count"],
            "missing_mask_count": phase3_evaluation["missing_mask_count"],
        }
    )
    model_path = ROOT / "artifacts" / category / "model.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"Baseline artifact is missing: {model_path}")
    return baseline_result, comparison_row["baseline_benchmark"], comparison_row


def _baseline_training_seconds(category: str, tracking_uri: str | None) -> float | None:
    if not tracking_uri:
        return None
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name("visionguard-visa")
        if experiment is None:
            return None
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{category}-train'",
            max_results=10,
        )
        finished = [run for run in runs if run.info.status == "FINISHED"]
        if not finished:
            return None
        return finished[0].data.metrics.get("training_seconds")
    except (ImportError, OSError, ValueError):
        return None


def _benchmark_candidate(
    model_path: Path,
    dataset_root: Path,
    category: str,
    warmup: int,
    samples: int,
) -> dict:
    from scripts.benchmark import _cpu_model, _process_memory_bytes

    load_start = perf_counter()
    model = MultiscalePatchCoreModel.load(model_path, device="cpu")
    model_load_seconds = perf_counter() - load_start
    items = load_test_items(dataset_root, category)
    image_paths = [item.image_path for item in items]
    if not image_paths:
        raise ValueError(f"No official test images found for benchmark: {category}")
    elapsed_ms: list[float] = []
    with MemoryMonitor() as monitor:
        for index in range(warmup + samples):
            path = image_paths[index % len(image_paths)]
            started = perf_counter()
            model.predict_image(path)
            duration = (perf_counter() - started) * 1000
            if index >= warmup:
                elapsed_ms.append(duration)
    memory_measurement = monitor.result()
    return {
        "status": "measured",
        "category": category,
        "dataset_identity": str(dataset_root.resolve()),
        "evaluation_split": "official test images",
        "device": "cpu",
        "warmup_count": warmup,
        "sample_count": samples,
        "model_load_seconds": model_load_seconds,
        "model_artifact_bytes": model_path.stat().st_size,
        "memory_bank_bytes": int(model.memory_bank.nbytes),
        "feature_grid": [model.input_size // 4, model.input_size // 4],
        "feature_dimension": int(model.memory_bank.shape[1]),
        "cpu_model": _cpu_model(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "timings": {
            "mean_ms": statistics.fmean(elapsed_ms),
            "median_ms": statistics.median(elapsed_ms),
            "p95_ms": float(np.percentile(elapsed_ms, 95)),
        },
        "throughput_images_per_second": 1000 / statistics.fmean(elapsed_ms),
        "cpu_peak_working_set_bytes": _process_memory_bytes(),
        "sampled_peak_working_set_bytes": memory_measurement[
            "observed_peak_working_set_bytes"
        ],
        "minimum_available_system_memory_bytes": memory_measurement[
            "minimum_available_system_memory_bytes"
        ],
        "cuda_available": torch.cuda.is_available(),
        "metric_scope": "end-to-end model.predict_image, excluding image decoding",
    }


def _run_benchmark_subprocess(
    model_path: Path, category: str, warmup: int, samples: int
) -> dict:
    output_path = OUTPUT_ROOT / category / "benchmark.json"
    command = [
        sys.executable,
        "-m",
        "scripts.experiment_multiscale",
        "--benchmark-only",
        "--model",
        str(model_path),
        "--category",
        category,
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Candidate benchmark failed ({completed.returncode}): "
            f"{completed.stderr[-4000:]}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _run_category(
    category: str,
    dataset_root: Path,
    tracking_uri: str | None,
    warmup: int,
    samples: int,
) -> dict:
    _, available_memory = _memory_snapshot()
    minimum_available = 6 * 1024**3
    if available_memory is not None and available_memory < minimum_available:
        raise MemoryError(
            f"Only {available_memory / 1024**3:.2f} GiB is available; "
            f"Phase 4 requires at least {minimum_available / 1024**3:.0f} GiB "
            "free before starting a category."
        )
    validate_dataset(dataset_root, category)
    baseline_path = ROOT / "artifacts" / category / "model.npz"
    output_dir = OUTPUT_ROOT / category
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.npz"
    config = _config_for_category(category, baseline_path, model_path)
    train_paths = load_train_images(dataset_root, category)
    config_path = OUTPUT_ROOT / "config.json"
    config_record = json.loads(config_path.read_text(encoding="utf-8"))

    fit_seconds = None
    fit_memory: dict[str, int | None] = {}
    if model_path.is_file():
        candidate = MultiscalePatchCoreModel.load(model_path, device="cpu")
        fit_metadata = candidate.fit_metadata
        metadata_path = output_dir / "model_metadata.json"
        if metadata_path.is_file():
            fit_memory = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "fit_memory", {}
            )
    else:
        candidate = MultiscalePatchCoreModel(
            config=config,
            projected_channels=int(
                config_record["feature_combination"]["per_scale_channels"]
            ),
            pretrained=True,
        )
        sample_tensor = preprocess_image(
            Image.open(train_paths[0]).convert("RGB"),
            size=candidate.input_size,
        )
        _, actual_grid = candidate.feature_extractor.extract(sample_tensor)
        if actual_grid != (56, 56):
            raise ValueError(f"Expected layer1 56x56 grid, observed {actual_grid}")
        started = perf_counter()
        with MemoryMonitor() as monitor:
            candidate.fit(train_paths)
        fit_memory = monitor.result()
        fit_seconds = perf_counter() - started
        fit_metadata = candidate.fit_metadata
        candidate.save(model_path)
        metadata_path = output_dir / "model_metadata.json"
        metadata = {
            "experiment_id": EXPERIMENT_ID,
            "category": category,
            "candidate_configuration": config_record,
            "fit_metadata": fit_metadata,
            "backbone_layer_shapes": candidate.feature_extractor.last_layer_shapes,
            "candidate_memory_bank_shape": list(candidate.memory_bank.shape),
            "candidate_memory_bank_bytes": int(candidate.memory_bank.nbytes),
            "artifact_size_bytes": model_path.stat().st_size,
            "fit_memory": fit_memory,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        _log_once(
            f"phase4-multiscale-{category}-training",
            {
                "experiment_id": EXPERIMENT_ID,
                "category": category,
                "dataset_identity": str(dataset_root.resolve()),
                "training_split": "official train/good only",
                "random_seed": candidate.random_seed,
                "configuration": json.dumps(config_record, sort_keys=True),
                "software_versions": json.dumps(_software_versions(), sort_keys=True),
            },
            {
                "training_seconds": fit_seconds,
                "memory_bank_patch_count": float(candidate.memory_bank.shape[0]),
                "memory_bank_dimension": float(candidate.memory_bank.shape[1]),
                "memory_bank_bytes": float(candidate.memory_bank.nbytes),
                "artifact_size_bytes": float(model_path.stat().st_size),
            },
            (model_path, metadata_path),
            tracking_uri,
        )

    calibration_path = output_dir / "calibration.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_seconds = calibration["calibration_seconds"]
        candidate.threshold = calibration["image_threshold"]
        candidate.pixel_threshold = calibration["pixel_threshold"]
    else:
        started = perf_counter()
        with MemoryMonitor() as monitor:
            image_threshold, pixel_threshold, calibration = _normal_calibration(
                candidate, train_paths
            )
        calibration_memory = monitor.result()
        calibration_seconds = perf_counter() - started
        candidate.threshold = image_threshold
        candidate.pixel_threshold = pixel_threshold
        calibration["calibration_seconds"] = calibration_seconds
        calibration["memory"] = calibration_memory
        calibration_path.write_text(
            json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
        )
        candidate.save(model_path)
        _log_once(
            f"phase4-multiscale-{category}-calibration",
            {
                "experiment_id": EXPERIMENT_ID,
                "category": category,
                "dataset_identity": str(dataset_root.resolve()),
                "calibration_data": "official normal train/good only",
                "random_seed": candidate.random_seed,
                "image_quantile": calibration["image_quantile"],
                "pixel_quantile": calibration["pixel_quantile"],
                "pixel_histogram_bins": HISTOGRAM_BINS,
            },
            {
                "calibration_seconds": calibration_seconds,
                "image_threshold": image_threshold,
                "pixel_threshold": pixel_threshold,
            },
            (calibration_path,),
            tracking_uri,
        )

    evaluation_path = output_dir / "evaluation.json"
    diagnostics_root = OUTPUT_ROOT / "diagnostics" / category
    baseline_result, baseline_benchmark, phase3_row = _baseline_data(category)
    reusable_evaluation = False
    if evaluation_path.is_file():
        saved_result = json.loads(evaluation_path.read_text(encoding="utf-8"))
        reusable_evaluation = (
            saved_result.get("status") == "measured"
            and saved_result.get("dataset_identity") == str(dataset_root.resolve())
            and saved_result.get("missing_mask_count") == 0
            and "diagnostics" in saved_result
        )
    if reusable_evaluation:
        result = saved_result
        evaluation_seconds = float(result["evaluation_seconds"])
        fit_seconds = result.get("fit_seconds", fit_seconds)
        evaluation_memory = result.get("evaluation_memory", {})
    else:
        started = perf_counter()
        with MemoryMonitor() as monitor:
            result = _evaluate_category_bounded(
                candidate,
                dataset_root,
                category,
                float(candidate.threshold),
                float(candidate.pixel_threshold),
            )
        evaluation_memory = monitor.result()
        evaluation_seconds = perf_counter() - started
        result.update(
            {
                "status": "measured",
                "experiment_id": EXPERIMENT_ID,
                "dataset_identity": str(dataset_root.resolve()),
                "evaluation_split": "official test",
                "evaluation_seconds": evaluation_seconds,
                "calibration_seconds": calibration_seconds,
                "fit_seconds": fit_seconds,
                "fit_metadata": fit_metadata,
                "fit_memory": fit_memory,
                "calibration_memory": calibration.get("memory", {}),
                "evaluation_memory": evaluation_memory,
                "model_artifact": str(model_path),
                "model_artifact_bytes": model_path.stat().st_size,
                "memory_bank_shape": list(candidate.memory_bank.shape),
                "memory_bank_bytes": int(candidate.memory_bank.nbytes),
            }
        )
        result["evaluation_memory"] = evaluation_memory
        evaluation_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

        candidate_diagnostics = _save_candidate_diagnostics(
            candidate,
            result,
            dataset_root,
            category,
            diagnostics_root / "candidate",
        )
        result["diagnostics"] = {
            "candidate": candidate_diagnostics,
            "baseline_directory": str(diagnostics_root / "baseline"),
        }
        evaluation_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    _copy_baseline_diagnostics(category, diagnostics_root / "baseline")

    benchmark_path = output_dir / "benchmark.json"
    reusable_benchmark = False
    if benchmark_path.is_file():
        saved_benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        reusable_benchmark = (
            saved_benchmark.get("status") == "measured"
            and saved_benchmark.get("warmup_count") == warmup
            and saved_benchmark.get("sample_count") == samples
        )
    benchmark = (
        saved_benchmark
        if reusable_benchmark
        else _run_benchmark_subprocess(
            model_path, category, warmup=warmup, samples=samples
        )
    )
    result["benchmark"] = benchmark
    result["baseline_model_artifact_bytes"] = (
        ROOT / "artifacts" / category / "model.npz"
    ).stat().st_size
    result["baseline_memory_bank_shape"] = [706, 1536]
    result["baseline_benchmark"] = baseline_benchmark
    result["baseline_metric_source"] = "Phase 3 bilinear baseline final test evaluation"
    result["baseline_metrics"] = {
        metric: baseline_result.get(metric) for metric in METRICS
    }
    evaluation_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    _log_once(
        f"phase4-multiscale-{category}-evaluation",
        {
            "experiment_id": EXPERIMENT_ID,
            "category": category,
            "dataset_identity": str(dataset_root.resolve()),
            "evaluation_split": "official test",
            "random_seed": candidate.random_seed,
            "image_threshold_source": "normal train/good only, exact 99th percentile",
            "pixel_threshold_source": "normal train/good only, streamed 99.5th percentile histogram",
            "feature_configuration": json.dumps(config_record, sort_keys=True),
        },
        {
            **{key: result.get(key) for key in METRICS},
            "evaluation_seconds": evaluation_seconds,
        },
        (evaluation_path, diagnostics_root),
        tracking_uri,
    )
    _log_once(
        f"phase4-multiscale-{category}-benchmark",
        {
            "experiment_id": EXPERIMENT_ID,
            "category": category,
            "dataset_identity": str(dataset_root.resolve()),
            "benchmark_split": "official test images",
            "device": "cpu",
            "warmup_count": warmup,
            "sample_count": samples,
            "random_seed": candidate.random_seed,
        },
        {
            "mean_latency_ms": benchmark["timings"]["mean_ms"],
            "median_latency_ms": benchmark["timings"]["median_ms"],
            "p95_latency_ms": benchmark["timings"]["p95_ms"],
            "throughput_images_per_second": benchmark[
                "throughput_images_per_second"
            ],
            "peak_working_set_bytes": benchmark["cpu_peak_working_set_bytes"],
        },
        (output_dir / "benchmark.json",),
        tracking_uri,
    )
    return {
        "category": category,
        "baseline": {metric: baseline_result.get(metric) for metric in METRICS},
        "candidate": {metric: result.get(metric) for metric in METRICS},
        "candidate_thresholds": {
            "image": candidate.threshold,
            "pixel": candidate.pixel_threshold,
        },
        "baseline_thresholds": {
            "image": baseline_result.get("image_threshold"),
            "pixel": baseline_result.get("pixel_threshold"),
        },
        "baseline_benchmark": baseline_benchmark,
        "candidate_benchmark": benchmark,
        "baseline_fit_seconds": _baseline_training_seconds(category, tracking_uri),
        "candidate_fit_seconds": fit_seconds,
        "baseline_calibration_seconds": (
            phase3_row.get("baseline_stage_durations") or {}
        ).get("calibration_seconds"),
        "candidate_calibration_seconds": calibration_seconds,
        "baseline_evaluation_seconds": (
            phase3_row.get("baseline_stage_durations") or {}
        ).get("evaluation_seconds"),
        "candidate_evaluation_seconds": evaluation_seconds,
        "baseline_artifact_bytes": result["baseline_model_artifact_bytes"],
        "candidate_artifact_bytes": model_path.stat().st_size,
        "baseline_memory_bank_shape": [706, 1536],
        "candidate_memory_bank_shape": list(candidate.memory_bank.shape),
        "baseline_memory_bank_bytes": baseline_benchmark["memory_bank_bytes"],
        "candidate_memory_bank_bytes": int(candidate.memory_bank.nbytes),
        "candidate_peak_cpu_memory_bytes": benchmark[
            "cpu_peak_working_set_bytes"
        ],
        "candidate_fit_peak_memory_bytes": fit_memory.get(
            "observed_peak_working_set_bytes"
        ),
        "candidate_calibration_peak_memory_bytes": result.get(
            "calibration_memory", {}
        ).get("observed_peak_working_set_bytes"),
        "candidate_evaluation_peak_memory_bytes": result.get(
            "evaluation_memory", {}
        ).get("observed_peak_working_set_bytes"),
        "candidate_evaluation_minimum_available_memory_bytes": result.get(
            "evaluation_memory", {}
        ).get("minimum_available_system_memory_bytes"),
        "candidate_benchmark_peak_memory_delta_bytes": (
            benchmark["cpu_peak_working_set_bytes"]
            - baseline_benchmark["cpu_peak_working_set_bytes"]
        ),
        "diagnostics": result["diagnostics"],
    }


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    for package in ("torchvision", "Pillow", "scikit-learn", "mlflow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _memory_snapshot() -> tuple[int | None, int | None]:
    if platform.system() != "Windows":
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            return peak, None
        except ImportError:
            return None, None

    import ctypes
    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError()
    return int(counters.WorkingSetSize), int(status.ullAvailPhys)


class MemoryMonitor:
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_working_set = 0
        self._minimum_available = math.inf

    def _sample(self) -> None:
        working_set, available = _memory_snapshot()
        if working_set is not None:
            self._peak_working_set = max(self._peak_working_set, working_set)
        if available is not None:
            self._minimum_available = min(self._minimum_available, available)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)
        self._sample()

    def __enter__(self) -> "MemoryMonitor":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def result(self) -> dict[str, int | None]:
        return {
            "observed_peak_working_set_bytes": self._peak_working_set or None,
            "minimum_available_system_memory_bytes": (
                int(self._minimum_available)
                if math.isfinite(self._minimum_available)
                else None
            ),
        }


def _write_comparison(rows: list[dict], config_record: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline_macro = macro_average(
        [
            {**row["baseline"], "category": row["category"]}
            for row in rows
        ]
    )
    candidate_macro = macro_average(
        [
            {**row["candidate"], "category": row["category"]}
            for row in rows
        ]
    )
    pixel_ap_delta = (
        candidate_macro.get("pixel_average_precision")
        - baseline_macro.get("pixel_average_precision")
        if candidate_macro.get("pixel_average_precision") is not None
        and baseline_macro.get("pixel_average_precision") is not None
        else None
    )
    image_recall_delta = (
        candidate_macro.get("image_recall") - baseline_macro.get("image_recall")
        if candidate_macro.get("image_recall") is not None
        and baseline_macro.get("image_recall") is not None
        else None
    )
    pixel_auroc_delta = (
        candidate_macro.get("pixel_auroc") - baseline_macro.get("pixel_auroc")
        if candidate_macro.get("pixel_auroc") is not None
        and baseline_macro.get("pixel_auroc") is not None
        else None
    )
    image_auroc_delta = (
        candidate_macro.get("image_auroc") - baseline_macro.get("image_auroc")
        if candidate_macro.get("image_auroc") is not None
        and baseline_macro.get("image_auroc") is not None
        else None
    )
    image_ap_delta = (
        candidate_macro.get("image_average_precision")
        - baseline_macro.get("image_average_precision")
        if candidate_macro.get("image_average_precision") is not None
        and baseline_macro.get("image_average_precision") is not None
        else None
    )
    pixel_ap_category_regressions = sum(
        1
        for row in rows
        if row["candidate"].get("pixel_average_precision") is not None
        and row["baseline"].get("pixel_average_precision") is not None
        and row["candidate"]["pixel_average_precision"]
        < row["baseline"]["pixel_average_precision"]
    )
    latency_within_budget = all(
        row["candidate_benchmark"]["timings"]["mean_ms"]
        <= 1.5
        * row["baseline_benchmark"]["timings"]["end_to_end_prediction"]["mean_ms"]
        for row in rows
    )
    if (
        len(rows) == len(CATEGORIES)
        and pixel_ap_delta is not None
        and pixel_ap_delta > 0.005
        and pixel_auroc_delta is not None
        and pixel_auroc_delta >= -0.005
        and image_recall_delta is not None
        and image_recall_delta >= -0.02
        and image_auroc_delta is not None
        and image_auroc_delta >= -0.01
        and image_ap_delta is not None
        and image_ap_delta >= -0.01
        and pixel_ap_category_regressions <= 1
        and latency_within_budget
    ):
        decision = (
            "promising on the three-category screen; expand to the remaining "
            "categories before any production promotion"
        )
    elif len(rows) == len(CATEGORIES):
        decision = "reject candidate and retain the production bilinear baseline"
    else:
        decision = "pending; comparison is partial and not suitable for promotion"
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured" if len(rows) == len(CATEGORIES) else "partial",
        "protocol": {
            "training": "official train/good only; no test labels or masks used",
            "calibration": (
                "candidate image thresholds: exact 99th percentile of normal "
                "train/good image scores; pixel thresholds: streamed 262144-bin "
                "histogram over bilinear maps of those same normal training images"
            ),
            "evaluation": "official test split only, after candidate fixed",
            "categories": [row["category"] for row in rows],
            "seed": config_record["random_seed"],
        },
        "candidate_configuration": config_record,
        "baseline_macro_average": baseline_macro,
        "candidate_macro_average": candidate_macro,
        "macro_delta_candidate_minus_baseline": {
            metric: (
                candidate_macro.get(metric) - baseline_macro.get(metric)
                if candidate_macro.get(metric) is not None
                and baseline_macro.get(metric) is not None
                else None
            )
            for metric in METRICS
        },
        "categories": rows,
        "promotion_decision": decision,
    }
    (OUTPUT_ROOT / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    columns = ["category"]
    columns.extend(f"baseline_{metric}" for metric in METRICS)
    columns.extend(f"candidate_{metric}" for metric in METRICS)
    columns.extend(f"delta_{metric}" for metric in METRICS)
    columns.extend(
        (
            "baseline_fit_seconds",
            "candidate_fit_seconds",
            "baseline_calibration_seconds",
            "candidate_calibration_seconds",
            "baseline_evaluation_seconds",
            "candidate_evaluation_seconds",
            "baseline_artifact_bytes",
            "candidate_artifact_bytes",
            "baseline_memory_bank_bytes",
            "candidate_memory_bank_bytes",
            "baseline_latency_mean_ms",
            "candidate_latency_mean_ms",
            "baseline_latency_median_ms",
            "candidate_latency_median_ms",
            "baseline_latency_p95_ms",
            "candidate_latency_p95_ms",
            "candidate_peak_cpu_memory_bytes",
            "candidate_benchmark_peak_memory_delta_bytes",
            "candidate_fit_peak_memory_bytes",
            "candidate_calibration_peak_memory_bytes",
            "candidate_evaluation_peak_memory_bytes",
            "candidate_evaluation_minimum_available_memory_bytes",
        )
    )
    with (OUTPUT_ROOT / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {"category": row["category"]}
            flat.update(
                {f"baseline_{key}": value for key, value in row["baseline"].items()}
            )
            flat.update(
                {f"candidate_{key}": value for key, value in row["candidate"].items()}
            )
            flat.update(
                {
                    f"delta_{metric}": (
                        row["candidate"].get(metric) - row["baseline"].get(metric)
                        if row["candidate"].get(metric) is not None
                        and row["baseline"].get(metric) is not None
                        else None
                    )
                    for metric in METRICS
                }
            )
            for key in (
                "baseline_fit_seconds",
                "candidate_fit_seconds",
                "baseline_calibration_seconds",
                "candidate_calibration_seconds",
                "baseline_evaluation_seconds",
                "candidate_evaluation_seconds",
                "baseline_artifact_bytes",
                "candidate_artifact_bytes",
                "baseline_memory_bank_bytes",
                "candidate_memory_bank_bytes",
                "candidate_peak_cpu_memory_bytes",
                "candidate_benchmark_peak_memory_delta_bytes",
                "candidate_fit_peak_memory_bytes",
                "candidate_calibration_peak_memory_bytes",
                "candidate_evaluation_peak_memory_bytes",
                "candidate_evaluation_minimum_available_memory_bytes",
            ):
                flat[key] = row.get(key)
            for kind in ("baseline", "candidate"):
                benchmark_timings = row[f"{kind}_benchmark"]["timings"]
                timings = (
                    benchmark_timings["end_to_end_prediction"]
                    if kind == "baseline"
                    else benchmark_timings
                )
                for stat in ("mean", "median", "p95"):
                    flat[f"{kind}_latency_{stat}_ms"] = timings[f"{stat}_ms"]
            writer.writerow(flat)

    report = [
        "# Phase 4 multiscale localization experiment",
        "",
        "Candidate: three-scale layer1/layer2/layer3 features aligned at 56x56. "
        "Each scale is projected to 128 seeded Gaussian channels, L2-normalized "
        "per location, weighted equally, and concatenated into 384-D patch vectors. "
        "Both baseline and candidate use bilinear anomaly-map resizing and max "
        "image-score aggregation.",
        "",
        "Training used only official `train/good` images. Candidate thresholds "
        "used normal training data only. Official test labels/masks were used "
        "only for final evaluation and diagnostics.",
        "",
        "## Per-category metrics (baseline → candidate)",
        "",
        "| Category | Image AUROC | Image AP | Image P/R/F1 | Pixel AUROC | Pixel AP | Pixel P/R/F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        before, after = row["baseline"], row["candidate"]
        report.append(
            f"| {row['category']} "
            f"| {before['image_auroc']:.4f} → {after['image_auroc']:.4f} "
            f"| {before['image_average_precision']:.4f} → {after['image_average_precision']:.4f} "
            f"| {before['image_precision']:.3f}/{before['image_recall']:.3f}/{before['image_f1']:.3f} "
            f"→ {after['image_precision']:.3f}/{after['image_recall']:.3f}/{after['image_f1']:.3f} "
            f"| {before['pixel_auroc']:.4f} → {after['pixel_auroc']:.4f} "
            f"| {before['pixel_average_precision']:.4f} → {after['pixel_average_precision']:.4f} "
            f"| {before['pixel_precision']:.3f}/{before['pixel_recall']:.3f}/{before['pixel_f1']:.3f} "
            f"→ {after['pixel_precision']:.3f}/{after['pixel_recall']:.3f}/{after['pixel_f1']:.3f} |"
        )
    report.extend(
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
        delta = payload["macro_delta_candidate_minus_baseline"][metric]
        if before is not None and after is not None:
            report.append(
                f"| {metric} | {before:.6f} | {after:.6f} | {delta:+.6f} |"
            )
    report.extend(
        [
            "",
            "Threshold-independent ranking metrics are AUROC and average precision. "
            "Precision, recall, and F1 are threshold-dependent and use only the "
            "separately calibrated normal-training thresholds.",
            "",
            "## Small-defect and category analysis",
            "",
        ]
    )
    for row in rows:
        before, after = row["baseline"], row["candidate"]
        report.append(
            f"- **{row['category']} pixel AP:** "
            f"{before['pixel_average_precision']:.4f} → "
            f"{after['pixel_average_precision']:.4f} "
            f"({after['pixel_average_precision'] - before['pixel_average_precision']:+.4f}); "
            f"image recall {before['image_recall']:.3f} → {after['image_recall']:.3f}."
        )
    report.extend(
        [
            "",
            "## Resources and runtime",
            "",
            "| Category | Bank shape baseline/candidate | Artifact bytes baseline/candidate | Fit s baseline/candidate | Calibration s baseline/candidate | Evaluation s baseline/candidate | Mean/median/p95 ms baseline/candidate | Candidate images/s | Peak working set bytes: baseline benchmark / candidate benchmark / candidate evaluation | Eval minimum available bytes | Benchmark peak delta bytes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        timings = row["candidate_benchmark"]["timings"]
        report.append(
            f"| {row['category']} | {row['baseline_memory_bank_shape']} / {row['candidate_memory_bank_shape']} "
            f"| {row['baseline_artifact_bytes']} / {row['candidate_artifact_bytes']} "
            f"| {row['baseline_fit_seconds'] if row['baseline_fit_seconds'] is not None else 'not logged'} / "
            f"{row['candidate_fit_seconds'] if row['candidate_fit_seconds'] is not None else 'loaded from prior run'} "
            f"| {row['baseline_calibration_seconds'] if row['baseline_calibration_seconds'] is not None else 'not recorded'} / "
            f"{row['candidate_calibration_seconds']:.2f} "
            f"| {row['baseline_evaluation_seconds'] if row['baseline_evaluation_seconds'] is not None else 'not recorded'} / "
            f"{row['candidate_evaluation_seconds']:.2f} "
            f"| {row['baseline_benchmark']['timings']['end_to_end_prediction']['mean_ms']:.2f}/"
            f"{row['baseline_benchmark']['timings']['end_to_end_prediction']['median_ms']:.2f}/"
            f"{row['baseline_benchmark']['timings']['end_to_end_prediction']['p95_ms']:.2f} / "
            f"{timings['mean_ms']:.2f}/{timings['median_ms']:.2f}/{timings['p95_ms']:.2f} "
            f"| {row['candidate_benchmark']['throughput_images_per_second']:.2f} "
            f"| {row['baseline_benchmark']['cpu_peak_working_set_bytes']} / "
            f"{row['candidate_peak_cpu_memory_bytes']} / "
            f"{row['candidate_evaluation_peak_memory_bytes']} "
            f"| {row['candidate_evaluation_minimum_available_memory_bytes']} "
            f"| {row['candidate_benchmark_peak_memory_delta_bytes']} |"
        )
    report.extend(
        [
            "",
            "## Diagnostics and interpretation",
            "",
            "Baseline and candidate TP/FN/FP diagnostics are stored under "
            "`diagnostics/<category>/{baseline,candidate}/`. Original, mask, "
            "heatmap, and overlay dimensions are recorded in metadata and checked "
            "against the source image.",
            "",
            "Candidate pixel threshold calibration uses a bounded-memory histogram; "
            "the maximum quantization error is recorded per category in "
            "`<category>/calibration.json`. Threshold-based P/R/F1 are therefore "
            "evaluated at the fixed normal-only threshold; AUROC and AP are "
            "threshold-independent.",
            "",
            f"Promotion screen decision: **{decision}**.",
            "",
            "Promotion decision is based on the measured metrics, category "
            "consistency, CPU latency, memory, and artifact size. The production "
            "baseline and running Docker deployment were not modified.",
        ]
    )
    (OUTPUT_ROOT / "evaluation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def _write_benchmark_summary(rows: list[dict]) -> None:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured" if len(rows) == len(CATEGORIES) else "partial",
        "device": "CPU",
        "warmup_count": 20,
        "sample_count_per_category": 100,
        "categories": [
            {
                "category": row["category"],
                "baseline": row["baseline_benchmark"],
                "candidate": row["candidate_benchmark"],
            }
            for row in rows
        ],
        "software_versions": _software_versions(),
        "limitations": [
            "Single host and CPU only; no GPU results.",
            "Baseline values are the recorded Phase 3 bilinear benchmark; candidate values use a fresh process.",
        ],
    }
    (OUTPUT_ROOT / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _verify_outputs() -> None:
    comparison = json.loads(
        (OUTPUT_ROOT / "comparison.json").read_text(encoding="utf-8")
    )
    if set(row["category"] for row in comparison["categories"]) != set(CATEGORIES):
        raise ValueError("Phase 4 comparison does not contain all three target categories")
    for row in comparison["categories"]:
        category = row["category"]
        report = json.loads(
            (OUTPUT_ROOT / category / "evaluation.json").read_text(encoding="utf-8")
        )
        if report["missing_mask_count"] != 0:
            raise ValueError(f"{category} evaluation lacks one or more test masks")
        candidate_cases = report["diagnostics"]["candidate"]
        diagnostic_dirs = [
            OUTPUT_ROOT / example
            for example in candidate_cases.values()
            if example is not None
        ]
        baseline_dir = Path(report["diagnostics"]["baseline_directory"])
        diagnostic_dirs.extend(
            baseline_dir / case
            for case in ("true_positive", "false_negative", "false_positive")
            if (baseline_dir / case).is_dir()
        )
        for case_dir in diagnostic_dirs:
            with Image.open(case_dir / "original.png") as original:
                source_size = original.size
            for name in ("ground_truth.png", "heatmap.png", "overlay.png"):
                with Image.open(case_dir / name) as image:
                    if image.size != source_size:
                        raise ValueError(
                            f"Misaligned diagnostic {case_dir / name}: {image.size} != {source_size}"
                        )


def _log_summary_run(tracking_uri: str | None) -> None:
    comparison_path = OUTPUT_ROOT / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    metrics = {
        f"macro_baseline_{key}": value
        for key, value in comparison["baseline_macro_average"].items()
        if value is not None
    }
    metrics.update(
        {
            f"macro_candidate_{key}": value
            for key, value in comparison["candidate_macro_average"].items()
            if value is not None
        }
    )
    metrics.update(
        {
            f"macro_delta_{key}": value
            for key, value in comparison[
                "macro_delta_candidate_minus_baseline"
            ].items()
            if value is not None
        }
    )
    _log_once(
        "phase4-multiscale-summary",
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_identity": comparison["candidate_configuration"][
                "dataset_identity"
            ],
            "training_split": "official train/good only",
            "calibration_split": "normal official train/good only",
            "evaluation_split": "official test",
            "random_seed": 42,
            "candidate_id": "layer1-layer2-layer3-proj128-equalnorm-v1",
            "promotion_decision": comparison["promotion_decision"],
        },
        metrics,
        (
            OUTPUT_ROOT / "config.json",
            OUTPUT_ROOT / "comparison.json",
            OUTPUT_ROOT / "comparison.csv",
            OUTPUT_ROOT / "evaluation_report.md",
            OUTPUT_ROOT / "benchmark_results.json",
            OUTPUT_ROOT / "diagnostics",
        ),
        tracking_uri,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated Phase 4 multiscale VisA experiment."
    )
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.benchmark_only:
        if args.model is None or args.category is None or args.output is None:
            parser.error("--benchmark-only requires --model, --category, and --output")
        config = _runtime_config()
        dataset_root = _dataset_path(config)
        result = _benchmark_candidate(
            args.model,
            dataset_root,
            args.category,
            args.warmup,
            args.samples,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
        return

    config = _runtime_config()
    dataset_root = _dataset_path(config)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Prepared VisA dataset not found at {dataset_root}")
    if args.warmup != 20 or args.samples != 100:
        print(
            f"Using requested benchmark protocol: warmup={args.warmup}, samples={args.samples}",
            flush=True,
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "experiment_id": EXPERIMENT_ID,
        "candidate_id": "layer1-layer2-layer3-proj128-equalnorm-v1",
        "dataset_identity": str(dataset_root.resolve()),
        "dataset": "official VisA prepared 1cls split",
        "categories": list(CATEGORIES),
        "training_split": "train/good",
        "calibration_split": "normal train/good only",
        "evaluation_split": "official test",
        "random_seed": 42,
        "device": "cpu",
        "image_size": 224,
        "baseline_image_score_aggregation": "max",
        "anomaly_map_upsampling": "bilinear",
        "feature_layers": {
            "layer1": {"original": [1, 256, 56, 56], "aligned": [1, 128, 56, 56]},
            "layer2": {"original": [1, 512, 28, 28], "aligned": [1, 128, 56, 56]},
            "layer3": {"original": [1, 1024, 14, 14], "aligned": [1, 128, 56, 56]},
        },
        "feature_combination": {
            "per_scale_channels": 128,
            "per_scale_projection": "seeded Gaussian random projection to 128 channels",
            "per_location_normalization": "L2 normalize each projected scale",
            "scale_weight": "1/sqrt(3)",
            "final_patch_grid": [56, 56],
            "final_channels": 384,
        },
        "memory_bank": {
            "sample_ratio": 0.001,
            "candidate_pool_cap": 6000,
            "candidate_sampling": "uniform seeded sample from all normal training patches",
            "selection": "greedy farthest point in seeded random 128-D projection",
        },
        "image_scoring": "maximum nearest-neighbor distance over 3136 patches",
        "threshold_quantiles": {"image": 0.99, "pixel": 0.995},
        "pixel_threshold_histogram_bins": HISTOGRAM_BINS,
        "benchmark": {"warmup_count": args.warmup, "sample_count": args.samples},
        "software_versions": _software_versions(),
    }
    config_path = OUTPUT_ROOT / "config.json"
    if config_path.exists():
        prior_config = json.loads(config_path.read_text(encoding="utf-8"))
        if prior_config != experiment_config:
            raise ValueError(
                f"Existing experiment config differs; refusing to overwrite {config_path}"
            )
    else:
        config_path.write_text(
            json.dumps(experiment_config, indent=2) + "\n", encoding="utf-8"
        )

    if torch.cuda.is_available():
        raise RuntimeError("Phase 4 must run on CPU; CUDA unexpectedly became available")

    rows = []
    for category in CATEGORIES:
        print(f"Starting Phase 4 category: {category}", flush=True)
        rows.append(
            _run_category(
                category,
                dataset_root,
                config.mlflow_tracking_uri,
                args.warmup,
                args.samples,
            )
        )
        _write_comparison(rows, experiment_config)
        _write_benchmark_summary(rows)
    _write_comparison(rows, experiment_config)
    _write_benchmark_summary(rows)
    _verify_outputs()
    _log_summary_run(config.mlflow_tracking_uri)
    print(
        f"Phase 4 completed: {OUTPUT_ROOT / 'evaluation_report.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
