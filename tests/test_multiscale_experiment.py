from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import models

from app.main import app
import scripts.experiment_multiscale as experiment
from scripts.experiment_multiscale import (
    MemoryMonitor,
    _evaluate_category_bounded,
    _exact_pixel_ranking_metrics,
    _radix_sort_scores,
)
from visionguard.data import build_synthetic_mvtec_dataset
from visionguard import experimental_multiscale
from visionguard.config import ModelConfig
from visionguard.evaluation import _binary_metrics
from visionguard.experimental_multiscale import (
    MultiscaleFeatureExtractor,
    MultiscalePatchCoreModel,
    combine_multiscale_maps,
)
from scripts.experiment_multiscale import _normal_calibration


def test_baseline_diagnostic_heatmap_is_regenerated_at_source_dimensions(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    source = (
        tmp_path
        / "reports"
        / "experiments"
        / "phase3_localization_v1"
        / "candle"
        / "examples"
        / "baseline"
        / "true_positive"
    )
    source.mkdir(parents=True)
    Image.new("RGB", (80, 40), (20, 30, 40)).save(source / "original.png")
    Image.new("L", (80, 40), 255).save(source / "ground_truth.png")
    Image.new("RGB", (40, 80)).save(source / "overlay.png")
    (source / "metadata.json").write_text(
        '{"kind":"true_positive"}', encoding="utf-8"
    )
    np.save(
        source / "raw_scores.npy",
        np.arange(20, dtype=np.float32).reshape(4, 5),
    )

    target = tmp_path / "phase4" / "baseline"
    experiment._copy_baseline_diagnostics("candle", target)

    for name in ("original.png", "ground_truth.png", "heatmap.png", "overlay.png"):
        with Image.open(target / "true_positive" / name) as image:
            assert image.size == (80, 40)
    metadata = json.loads(
        (target / "true_positive" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["source_dimensions"] == {"width": 80, "height": 40}
    assert metadata["heatmap"] == str(
        (target / "true_positive" / "heatmap.png").relative_to(tmp_path)
    )


def test_multiscale_alignment_projection_and_scale_normalization():
    layer1 = torch.ones((1, 4, 8, 8), dtype=torch.float32)
    layer2 = torch.ones((1, 6, 4, 4), dtype=torch.float32)
    layer3 = torch.ones((1, 8, 2, 2), dtype=torch.float32)

    first = combine_multiscale_maps(
        layer1, layer2, layer3, projected_channels=3
    )
    second = combine_multiscale_maps(
        layer1, layer2, layer3, projected_channels=3
    )

    assert first.shape == (64, 9)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for start in (0, 3, 6):
        scale_norms = torch.linalg.vector_norm(first[:, start : start + 3], dim=1)
        torch.testing.assert_close(
            scale_norms,
            torch.full((64,), 1 / np.sqrt(3), dtype=torch.float32),
        )


def test_multiscale_feature_extraction_is_deterministic_and_56_by_56(monkeypatch):
    def untrained_backbone(model_name: str, pretrained: bool):
        assert model_name == "wide_resnet50_2"
        assert pretrained is False
        torch.manual_seed(123)
        return models.wide_resnet50_2(weights=None).eval()

    monkeypatch.setattr(experimental_multiscale, "_load_backbone", untrained_backbone)
    extractor = MultiscaleFeatureExtractor(pretrained=False, random_seed=42)
    tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    features1, shape1 = extractor.extract(tensor)
    observed_shapes = dict(extractor.last_layer_shapes)
    features2, shape2 = extractor.extract(tensor)

    assert shape1 == shape2 == (56, 56)
    assert observed_shapes == {
        "layer1": (1, 256, 56, 56),
        "layer2": (1, 512, 28, 28),
        "layer3": (1, 1024, 14, 14),
    }
    assert features1.shape == (56 * 56, 384)
    np.testing.assert_array_equal(features1, features2)


def test_multiscale_memory_bank_is_seeded_and_dimensionally_valid(tmp_path):
    class DeterministicExtractor:
        def extract(self, _tensor):
            features = np.arange(4 * 384, dtype=np.float32).reshape(4, 384)
            return features, (2, 2)

    image_paths = []
    for index in range(3):
        path = tmp_path / f"normal-{index}.png"
        Image.new("RGB", (8, 8), (index, index, index)).save(path)
        image_paths.append(path)

    models_fitted = []
    for _ in range(2):
        model = MultiscalePatchCoreModel.__new__(MultiscalePatchCoreModel)
        model.config = ModelConfig(image_size=8, sample_ratio=0.5, random_seed=11)
        model.input_size = 8
        model.sample_ratio = 0.5
        model.random_seed = 11
        model.projected_channels = 128
        model.feature_extractor = DeterministicExtractor()
        model.memory_bank = None
        model.threshold = None
        model.pixel_threshold = None
        model.fit(image_paths)
        models_fitted.append(model.memory_bank)

    assert models_fitted[0].shape == (6, 384)
    np.testing.assert_array_equal(models_fitted[0], models_fitted[1])


def test_multiscale_model_serialization_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experimental_multiscale,
        "_load_backbone",
        lambda _name, pretrained: torch.nn.Linear(1, 1),
    )
    config = ModelConfig(
        category="candle",
        image_size=32,
        sample_ratio=0.25,
        random_seed=5,
        model_name="wide_resnet50_2",
    )
    model = MultiscalePatchCoreModel(
        config=config,
        projected_channels=2,
        pretrained=False,
    )
    model.memory_bank = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    model.threshold = 0.75
    model.pixel_threshold = 0.25
    artifact = tmp_path / "phase4-model.npz"
    model.save(artifact)

    loaded = MultiscalePatchCoreModel.load(artifact)

    np.testing.assert_array_equal(loaded.memory_bank, model.memory_bank)
    assert loaded.threshold == model.threshold
    assert loaded.pixel_threshold == model.pixel_threshold
    assert loaded.projected_channels == 2
    for index in range(3):
        torch.testing.assert_close(
            loaded.feature_extractor.projections[index],
            model.feature_extractor.projections[index],
            rtol=0,
            atol=0,
        )


def test_multiscale_calibration_reads_only_supplied_normal_paths(tmp_path):
    train_path = tmp_path / "train-good.png"
    Image.new("RGB", (6, 4), (1, 2, 3)).save(train_path)
    seen: list[Path] = []

    class CalibrationProbe:
        def predict_patch_map(self, path):
            seen.append(Path(path))
            return (
                1.5,
                np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
                (4, 6),
            )

    image_threshold, pixel_threshold, details = _normal_calibration(
        CalibrationProbe(), [train_path]
    )

    assert seen == [train_path]
    assert image_threshold == 1.5
    assert details["normal_image_count"] == 1
    assert details["calibration_data"] == "official normal train/good only"
    assert details["pixel_count"] == 24
    assert 0.0 <= pixel_threshold <= 3.0


def test_existing_pixel_metric_definitions_remain_unchanged():
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)

    measured = _binary_metrics(labels, scores, threshold=0.5, prefix="pixel")

    assert measured["pixel_confusion_matrix"] == [[2, 0], [1, 1]]
    assert measured["pixel_auroc"] == pytest.approx(0.75)
    assert measured["pixel_precision"] == 1.0
    assert measured["pixel_recall"] == 0.5
    assert measured["pixel_f1"] == pytest.approx(2 / 3)


def test_external_pixel_ranking_matches_existing_metrics_with_ties(tmp_path):
    values = np.array([0.1, 0.2, 0.2, 0.8, 0.8, 0.0], dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 0, 0], dtype=np.uint8)
    score_map = np.memmap(
        tmp_path / "scores.bin", mode="w+", dtype=np.float32, shape=values.shape
    )
    label_map = np.memmap(
        tmp_path / "labels.bin", mode="w+", dtype=np.uint8, shape=labels.shape
    )
    score_map[:] = values
    label_map[:] = labels
    score_map.flush()
    label_map.flush()

    sorted_keys, sorted_labels = _radix_sort_scores(
        score_map, label_map, tmp_path, chunk_size=2
    )
    observed_auc, observed_ap, status = _exact_pixel_ranking_metrics(
        sorted_keys, sorted_labels, values.size, chunk_size=2
    )
    expected = _binary_metrics(labels, values, threshold=0.5, prefix="pixel")

    assert status.startswith("computed exactly")
    assert observed_auc == pytest.approx(expected["pixel_auroc"])
    assert observed_ap == pytest.approx(expected["pixel_average_precision"])
    for memory_map in (sorted_keys, sorted_labels, score_map, label_map):
        memory_map.flush()
        memory_map._mmap.close()


def test_bounded_multiscale_evaluation_preserves_full_resolution_metrics(tmp_path):
    category_root = build_synthetic_mvtec_dataset(
        tmp_path / "visa",
        category="tile",
        train_good=2,
        test_good=1,
        test_bad=1,
        image_size=64,
    )

    class DeterministicExtractor:
        def extract(self, _tensor):
            vectors = np.array(
                [[0.0] * 6, [0.1] * 6, [0.2] * 6, [0.3] * 6],
                dtype=np.float32,
            )
            return vectors, (2, 2)

    model = MultiscalePatchCoreModel.__new__(MultiscalePatchCoreModel)
    model.config = ModelConfig(image_size=32)
    model.input_size = 32
    model.projected_channels = 2
    model.feature_extractor = DeterministicExtractor()
    model.memory_bank = np.zeros((2, 6), dtype=np.float32)
    model.threshold = 0.5
    model.pixel_threshold = 0.4

    with MemoryMonitor() as monitor:
        result = _evaluate_category_bounded(
            model,
            category_root.parent,
            "tile",
            threshold=0.5,
            pixel_threshold=0.4,
        )
    assert monitor.result()["observed_peak_working_set_bytes"] is not None
    assert result["test_image_count"] == 2
    assert result["missing_mask_count"] == 0
    assert result["pixel_count"] == 64 * 64 * 2
    assert result["pixel_auroc"] is not None
    assert result["pixel_average_precision"] is not None
    assert result["pixel_metric_implementation"].startswith("exact existing")


def test_multiscale_predictor_remains_api_compatible(monkeypatch, tmp_path):
    class LightweightMultiscaleExtractor:
        def extract(self, _tensor):
            return np.ones((4, 6), dtype=np.float32), (2, 2)

    model = MultiscalePatchCoreModel.__new__(MultiscalePatchCoreModel)
    model.config = ModelConfig(image_size=32)
    model.input_size = 32
    model.projected_channels = 2
    model.feature_extractor = LightweightMultiscaleExtractor()
    model.memory_bank = np.zeros((2, 6), dtype=np.float32)
    model.threshold = 0.5
    model.pixel_threshold = 0.5

    monkeypatch.setattr(app.state, "model", model)
    monkeypatch.setattr(app.state.config, "heatmap_dir", str(tmp_path))
    image_buffer = tmp_path / "request.png"
    Image.new("RGB", (24, 16), (120, 140, 160)).save(image_buffer)

    with image_buffer.open("rb") as stream, TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("request.png", stream.read(), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_dimensions"] == {"width": 24, "height": 16}
    assert set(payload["heatmap_artifacts"]) == {
        "original",
        "heatmap",
        "overlay",
        "legend",
    }
