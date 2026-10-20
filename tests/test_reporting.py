from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.benchmark import _summary
from scripts.evaluate import METRIC_COLUMNS, _write_reports
from scripts.experiment_localization import (
    _determine_nearest_pixel_threshold,
    _save_localization_diagnostics,
)
from scripts.generate_status_reports import _existing_verification_record
from scripts.validate_dataset import build_manifest
from visionguard import evaluation
from visionguard.evaluation import _binary_metrics
from visionguard.evaluation import weighted_quantile
from visionguard.data import DatasetItem, build_synthetic_mvtec_dataset


def test_dataset_manifest_validates_selected_category_and_records_splits(tmp_path):
    build_synthetic_mvtec_dataset(
        tmp_path / "dataset",
        category="tile",
        train_good=2,
        test_good=1,
        test_bad=1,
    )
    manifest = build_manifest(tmp_path / "dataset", category="tile")
    assert manifest["valid"] is True
    assert manifest["scope"] == "tile"
    assert manifest["image_counts"]["tile"]["train_good"] == 2
    assert manifest["image_counts"]["tile"]["test_good"] == 1
    assert manifest["image_counts"]["tile"]["test_anomalous"] == 1
    assert manifest["image_counts"]["tile"]["masks_found"] == 1


def test_benchmark_summary_reports_requested_latency_percentiles():
    result = _summary([1.0, 2.0, 3.0, 4.0])
    assert result["mean_ms"] == 2.5
    assert result["median_ms"] == 2.5
    np.testing.assert_allclose(result["p90_ms"], 3.7)
    np.testing.assert_allclose(result["p95_ms"], 3.85)


def test_threshold_calibration_is_passed_normal_train_paths_only(monkeypatch):
    normal_paths = [Path("train/good/a.png"), Path("train/good/b.png")]
    test_items = [
        DatasetItem(Path("test/good/normal.png"), None, "good", "tile"),
        DatasetItem(Path("test/crack/defect.png"), None, "crack", "tile"),
    ]
    monkeypatch.setattr(evaluation, "validate_dataset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluation, "load_train_images", lambda *_args: normal_paths)
    monkeypatch.setattr(evaluation, "load_test_items", lambda *_args: test_items)

    threshold_calls = []

    def image_threshold(paths, _model):
        threshold_calls.append(list(paths))
        return 0.2

    def pixel_threshold(paths, _model):
        threshold_calls.append(list(paths))
        return 0.3

    monkeypatch.setattr(evaluation, "determine_threshold", image_threshold)
    monkeypatch.setattr(evaluation, "determine_pixel_threshold", pixel_threshold)

    class Model:
        def predict_image(self, _path):
            return 0.1, np.zeros((2, 2), dtype=np.float32)

    result = evaluation.evaluate_category(Model(), "unused", "tile")
    assert threshold_calls == [normal_paths, normal_paths]
    assert result["image_threshold"] == 0.2
    assert result["pixel_threshold"] == 0.3
    assert result["pixel_auroc"] is None
    assert result["missing_mask_count"] == 1


def test_binary_metrics_keep_standard_threshold_and_ranking_definitions():
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.8, 0.7, 0.9], dtype=np.float64)
    result = _binary_metrics(labels, scores, threshold=0.75, prefix="image")

    assert result["image_confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["image_precision"] == 0.5
    assert result["image_recall"] == 0.5
    assert result["image_f1"] == 0.5
    assert result["image_auroc"] == 0.75
    assert result["image_average_precision"] == 0.8333333333333333


def test_pixel_metrics_use_aligned_binary_masks_and_standard_definitions(
    tmp_path, monkeypatch
):
    category_root = build_synthetic_mvtec_dataset(
        tmp_path / "dataset",
        category="tile",
        train_good=1,
        test_good=1,
        test_bad=1,
        image_size=8,
    )
    mask_path = (
        category_root
        / "ground_truth"
        / "synthetic_defect"
        / "defective_00_mask.png"
    )
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4, :4] = 255
    Image.fromarray(mask).save(mask_path)

    class FixedMapModel:
        def predict_image(self, path):
            if Path(path).parent.name == "synthetic_defect":
                return 1.0, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
            return 0.0, np.zeros((2, 2), dtype=np.float32)

    result = evaluation.evaluate_category(
        FixedMapModel(),
        category_root.parent,
        "tile",
        threshold=0.5,
        pixel_threshold=0.5,
    )

    assert result["missing_mask_count"] == 0
    assert result["pixel_confusion_matrix"] == [[7, 0], [0, 1]]
    assert result["pixel_auroc"] == 1.0
    assert result["pixel_average_precision"] == 1.0
    assert result["pixel_precision"] == 1.0
    assert result["pixel_recall"] == 1.0
    assert result["pixel_f1"] == 1.0


def test_weighted_quantile_matches_repeated_full_resolution_values():
    values = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    weights = np.array([2, 3, 1], dtype=np.int64)
    expanded = np.repeat(values, weights)

    for quantile in (0.0, 0.5, 0.8, 0.995, 1.0):
        assert weighted_quantile(values, weights, quantile) == np.quantile(
            expanded, quantile
        )


def test_candidate_pixel_calibration_matches_full_nearest_resized_maps():
    patch_map = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    expanded_map = np.repeat(np.repeat(patch_map, 4, axis=0), 4, axis=1)

    class FixedPatchMapModel:
        def predict_patch_map(self, path):
            assert path == Path("train/good/normal.png")
            return 0.0, patch_map, (8, 8)

    threshold = _determine_nearest_pixel_threshold(
        [Path("train/good/normal.png")],
        FixedPatchMapModel(),
        quantile=0.995,
    )

    assert threshold == np.quantile(expanded_map.reshape(-1), 0.995)


def test_false_positive_localization_diagnostic_saves_empty_ground_truth_mask(
    tmp_path,
):
    category_root = build_synthetic_mvtec_dataset(
        tmp_path / "dataset",
        category="tile",
        train_good=1,
        test_good=1,
        test_bad=1,
        image_size=8,
    )
    from visionguard.data import load_test_items

    good_item = next(
        item
        for item in load_test_items(category_root.parent, "tile")
        if item.label == "good"
    )

    class FixedMapModel:
        def predict_image(self, _path):
            return 1.0, np.ones((8, 8), dtype=np.float32)

    examples = _save_localization_diagnostics(
        FixedMapModel(),
        {
            "image_predictions": [
                {
                    "image_path": str(good_item.image_path),
                    "label": 0,
                    "prediction": 1,
                    "threshold": 0.5,
                }
            ]
        },
        category_root.parent,
        "tile",
        tmp_path / "examples",
    )
    false_positive = examples[0]
    assert false_positive["kind"] == "false_positive"
    with Image.open(false_positive["ground_truth_mask"]) as ground_truth:
        mask = np.asarray(ground_truth)
    assert mask.shape == (8, 8)
    assert not mask.any()


def test_missing_dataset_manifest_does_not_claim_download_or_metrics(tmp_path):
    manifest = build_manifest(tmp_path / "absent", category="bottle")
    assert manifest["valid"] is False
    assert manifest["download_status"] == "not downloaded"
    assert manifest["terms_status"].startswith("not verified")
    assert manifest["image_counts"] == {}


def test_dockerfile_declares_non_root_user_and_health_check():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "USER 10001:10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "VISIONGUARD_MODEL_PATH=/app/artifacts/model.npz" in dockerfile


def test_evaluation_reports_store_per_image_predictions_and_null_metrics(tmp_path):
    result = {
        "category": "bottle",
        "dataset_type": "MVTec AD",
        "image_threshold": 0.3,
        "pixel_threshold": 0.4,
        "test_image_count": 1,
        "missing_mask_count": 1,
        "image_predictions": [
            {
                "image_path": "test/crack/example.png",
                "label": 1,
                "defect_category": "crack",
                "score": 0.8,
                "threshold": 0.3,
                "prediction": 1,
                "mask_path": None,
            }
        ],
        **{key: None for key in METRIC_COLUMNS},
    }
    _write_reports([result], [], tmp_path)
    assert (tmp_path / "evaluation_results.json").is_file()
    assert (tmp_path / "per_category_metrics.csv").is_file()
    assert (tmp_path / "image_predictions.csv").is_file()
    assert "null" in (tmp_path / "evaluation_results.json").read_text(encoding="utf-8")
    prediction_csv = (tmp_path / "image_predictions.csv").read_text(encoding="utf-8")
    assert "test/crack/example.png" in prediction_csv


def test_evaluation_report_marks_unfinished_categories_as_partial(tmp_path):
    result = {
        "category": "bottle",
        "dataset_type": "VisA",
        "test_image_count": 2,
        "missing_mask_count": 0,
        "image_predictions": [],
        **{key: None for key in METRIC_COLUMNS},
    }
    _write_reports([result], [], tmp_path, requested_categories=["bottle", "cable"])
    payload = json.loads(
        (tmp_path / "evaluation_results.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "partial"
    assert payload["pending_categories"] == ["cable"]
    assert "cable" in (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")


def test_status_report_generator_preserves_verified_validation_record(tmp_path):
    path = tmp_path / "validation_log.md"
    path.write_text(
        "# Validation log\n\n## Verification record\n\nPytest passed: 39 tests.\n",
        encoding="utf-8",
    )
    assert _existing_verification_record(path) == (
        "## Verification record\n\nPytest passed: 39 tests."
    )
