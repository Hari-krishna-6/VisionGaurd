from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import models

from visionguard import features
from visionguard.data import (
    build_synthetic_mvtec_dataset,
    load_test_items,
    validate_dataset,
)
from visionguard.evaluation import evaluate_category
from visionguard.features import (
    FeatureExtractor,
    nearest_resize_pixel_counts,
    preprocess_image,
    resize_anomaly_map,
)
from visionguard.memory_bank import MemoryBank
from visionguard import memory_bank as memory_bank_module
from visionguard.config import ModelConfig
from visionguard.model import PatchCoreModel
from visionguard.visualization import generate_heatmap, save_heatmap_artifacts


@pytest.fixture
def untrained_backbone(monkeypatch):
    pretrained_requests = []

    def load_backbone(model_name="wide_resnet50_2", pretrained=True):
        assert model_name == "wide_resnet50_2"
        assert isinstance(pretrained, bool)
        pretrained_requests.append(pretrained)
        torch.manual_seed(123)
        return models.wide_resnet50_2(weights=None).eval()

    monkeypatch.setattr(features, "_load_backbone", load_backbone)
    return pretrained_requests


def test_validate_dataset_accepts_synthetic_dataset(tmp_path):
    root = build_synthetic_mvtec_dataset(tmp_path / "mvtec", category="tile")
    assert validate_dataset(tmp_path / "mvtec", category="tile") == ["tile"]
    assert (root / "train" / "good").exists()


def test_validate_dataset_rejects_missing_train_split(tmp_path):
    bad = tmp_path / "broken"
    (bad / "tile" / "test" / "good").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        validate_dataset(bad, category="tile")


def test_preprocess_image_returns_expected_shape():
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    assert preprocess_image(arr, size=32).shape == (1, 3, 32, 32)


@pytest.mark.usefixtures("untrained_backbone")
def test_feature_extraction_combines_intermediate_maps_and_is_deterministic():
    extractor = FeatureExtractor(device="cpu")
    input_tensor = torch.randn(1, 3, 64, 96)
    features1, spatial_shape = extractor.extract(input_tensor)
    features2, second_shape = extractor.extract(input_tensor)

    assert spatial_shape == second_shape == (8, 12)
    assert features1.shape == (8 * 12, 512 + 1024)
    np.testing.assert_array_equal(features1, features2)


@pytest.mark.usefixtures("untrained_backbone")
def test_default_224_input_produces_28_by_28_1536_channel_patch_grid():
    extractor = FeatureExtractor(device="cpu", pretrained=False)
    features, spatial_shape = extractor.extract(torch.zeros(1, 3, 224, 224))

    assert spatial_shape == (28, 28)
    assert features.shape == (784, 1536)


def test_memory_bank_uses_reproducible_greedy_coreset_and_chunked_distances():
    features = np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float32)
    selected1 = MemoryBank(features, sample_ratio=0.5, random_seed=7).coreset_subsample()
    selected2 = MemoryBank(features, sample_ratio=0.5, random_seed=7).coreset_subsample()
    np.testing.assert_array_equal(selected1, selected2)
    assert selected1[:, 0].min() == 0
    assert selected1[:, 0].max() == 10

    bank = MemoryBank(np.array([[0.0], [10.0]], dtype=np.float32))
    distances = bank.nearest_distances(
        np.array([[1.0], [8.0], [14.0]], dtype=np.float32),
        query_chunk_size=1,
        bank_chunk_size=1,
    )
    np.testing.assert_allclose(distances, [1.0, 2.0, 4.0])


def test_greedy_coreset_candidate_pool_and_projection_are_bounded(monkeypatch):
    monkeypatch.setattr(memory_bank_module, "MAX_GREEDY_CANDIDATES", 12)
    monkeypatch.setattr(memory_bank_module, "MAX_GREEDY_DIMENSIONS", 4)
    features = np.random.default_rng(0).normal(size=(20, 6)).astype(np.float32)
    first = MemoryBank(features, sample_ratio=0.25, random_seed=11).coreset_subsample()
    second = MemoryBank(features, sample_ratio=0.25, random_seed=11).coreset_subsample()
    assert first.shape == (5, 6)
    np.testing.assert_array_equal(first, second)


@pytest.mark.usefixtures("untrained_backbone")
def test_model_uses_real_feature_grid_and_returns_original_dimensions(tmp_path):
    category_root = build_synthetic_mvtec_dataset(
        tmp_path / "mvtec", category="tile", train_good=2, test_good=1, test_bad=1
    )
    model = PatchCoreModel(input_size=64, sample_ratio=0.25, random_seed=7, device="cpu")
    model.fit(sorted((category_root / "train" / "good").iterdir()))

    source = Image.new("RGB", (48, 32), color=(120, 140, 160))
    score, anomaly_map = model.predict_image(source)
    assert isinstance(score, float)
    assert anomaly_map.shape == (32, 48)
    assert np.isfinite(anomaly_map).all()


def test_anomaly_map_upsampling_preserves_requested_image_shape_and_grid_values():
    patch_map = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    nearest = resize_anomaly_map(patch_map, (6, 8), interpolation="nearest")
    bilinear = resize_anomaly_map(patch_map, (6, 8), interpolation="bilinear")

    assert nearest.shape == bilinear.shape == (6, 8)
    assert set(np.unique(nearest)) == {1.0, 2.0, 3.0, 4.0}
    assert not np.array_equal(nearest, bilinear)
    np.testing.assert_array_equal(
        nearest,
        resize_anomaly_map(patch_map, (6, 8), interpolation="nearest"),
    )
    with pytest.raises(ValueError, match="Unsupported anomaly-map interpolation"):
        resize_anomaly_map(patch_map, (6, 8), interpolation="cubic")


def test_nearest_upsample_pixel_counts_cover_every_source_pixel():
    counts = nearest_resize_pixel_counts((2, 2), (6, 8))

    assert counts.shape == (4,)
    assert counts.sum() == 6 * 8
    assert (counts > 0).all()


@pytest.mark.usefixtures("untrained_backbone")
def test_topk_mean_image_score_suppresses_single_patch_outlier():
    model = PatchCoreModel(
        config=ModelConfig(
            image_score_aggregation="topk_mean",
            image_score_top_k_fraction=0.5,
        ),
        pretrained=False,
    )
    distances = np.array([1.0, 1.0, 1.0, 9.0], dtype=np.float32)
    assert model._aggregate_image_score(distances) == 5.0
    model.config.image_score_aggregation = "max"
    assert model._aggregate_image_score(distances) == 9.0


@pytest.mark.usefixtures("untrained_backbone")
def test_topk_mean_configuration_is_validated_and_persisted(tmp_path):
    with pytest.raises(ValueError, match="image_score_top_k_fraction"):
        ModelConfig(image_score_top_k_fraction=0)
    with pytest.raises(ValueError, match="image_score_aggregation"):
        ModelConfig(image_score_aggregation="median")

    model = PatchCoreModel(
        config=ModelConfig(
            image_score_aggregation="topk_mean",
            image_score_top_k_fraction=0.02,
        ),
        input_size=32,
        sample_ratio=0.5,
        device="cpu",
        pretrained=False,
    )
    model.memory_bank = np.random.default_rng(0).normal(size=(10, 32)).astype(np.float32)
    artifact = tmp_path / "candidate.npz"
    model.save(artifact)
    loaded = PatchCoreModel.load(artifact)
    assert loaded.config.image_score_aggregation == "topk_mean"
    assert loaded.config.image_score_top_k_fraction == 0.02


@pytest.mark.usefixtures("untrained_backbone")
def test_anomaly_map_upsampling_configuration_is_validated_and_persisted(tmp_path):
    with pytest.raises(ValueError, match="anomaly_map_upsampling"):
        ModelConfig(anomaly_map_upsampling="cubic")

    model = PatchCoreModel(
        config=ModelConfig(anomaly_map_upsampling="nearest"),
        input_size=32,
        sample_ratio=0.5,
        device="cpu",
        pretrained=False,
    )
    model.memory_bank = np.random.default_rng(0).normal(size=(10, 32)).astype(np.float32)
    artifact = tmp_path / "nearest-model.npz"
    model.save(artifact)

    loaded = PatchCoreModel.load(artifact)
    assert loaded.config.anomaly_map_upsampling == "nearest"


def test_mvtec_defect_category_and_mask_path_are_loaded(tmp_path):
    category_root = build_synthetic_mvtec_dataset(
        tmp_path / "mvtec", category="tile", train_good=1, test_good=1, test_bad=1
    )
    (category_root / "test" / "synthetic_defect").rename(category_root / "test" / "crack")
    (
        category_root / "ground_truth" / "synthetic_defect"
    ).rename(category_root / "ground_truth" / "crack")
    items = load_test_items(tmp_path / "mvtec", "tile")
    anomaly_item = next(item for item in items if item.label != "good")
    assert anomaly_item.mask_path == (
        category_root / "ground_truth" / "crack" / "defective_00_mask.png"
    )
    assert anomaly_item.label == "crack"
    assert anomaly_item.mask is not None


@pytest.mark.usefixtures("untrained_backbone")
def test_evaluation_reports_image_and_pixel_metrics_and_missing_masks(tmp_path):
    category_root = build_synthetic_mvtec_dataset(
        tmp_path / "mvtec", category="tile", train_good=2, test_good=1, test_bad=1
    )
    dataset_root = category_root.parent
    model = PatchCoreModel(input_size=64, sample_ratio=0.25, device="cpu")
    model.fit(sorted((category_root / "train" / "good").iterdir()))
    results = evaluate_category(model, dataset_root, "tile")
    assert results["image_auroc"] is not None
    assert results["image_average_precision"] is not None
    assert results["image_precision"] is not None
    assert results["image_recall"] is not None
    assert results["image_f1"] is not None
    assert results["image_false_positive_rate"] is not None
    assert results["image_false_negative_rate"] is not None
    assert results["pixel_auroc"] is not None
    assert results["pixel_average_precision"] is not None
    assert results["pixel_precision"] is not None
    assert results["pixel_recall"] is not None
    assert results["pixel_f1"] is not None
    assert results["pixel_false_positive_rate"] is not None
    assert results["pixel_false_negative_rate"] is not None
    assert len(results["confusion_matrix"]) == 2
    assert len(results["pixel_confusion_matrix"]) == 2

    (category_root / "ground_truth" / "synthetic_defect" / "defective_00_mask.png").unlink()
    results_without_mask = evaluate_category(model, dataset_root, "tile")
    assert results_without_mask["pixel_auroc"] is None
    assert results_without_mask["pixel_average_precision"] is None
    assert results_without_mask["pixel_confusion_matrix"] is None
    assert results_without_mask["missing_mask_count"] == 1


def test_heatmap_normalizes_without_changing_score_map_and_saves_artifacts(tmp_path):
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
    )
    raw_map = np.linspace(5.0, 10.0, num=32 * 48, dtype=np.float32).reshape(32, 48)
    original = raw_map.copy()
    heatmap = generate_heatmap(raw_map, image.size[::-1])
    output = save_heatmap_artifacts(image, raw_map, tmp_path, stem="demo")
    np.testing.assert_array_equal(raw_map, original)
    assert heatmap.shape == (32, 48, 3)
    assert set(output) == {"original", "heatmap", "overlay", "legend"}
    assert all((tmp_path / path.split("\\")[-1]).is_file() for path in output.values())


def test_model_save_and_load_round_trip(tmp_path, untrained_backbone):
    model = PatchCoreModel(input_size=32, sample_ratio=0.5, random_seed=7, device="cpu")
    model.memory_bank = np.random.default_rng(0).normal(size=(10, 32)).astype(np.float32)
    model.threshold = 1.0
    model.pixel_threshold = 2.0
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = PatchCoreModel.load(path, device="cpu")
    assert loaded.memory_bank is not None
    assert loaded.threshold == 1.0
    assert loaded.pixel_threshold == 2.0
    np.testing.assert_array_equal(loaded.memory_bank, model.memory_bank)
    for name, tensor in model.feature_extractor.model.state_dict().items():
        torch.testing.assert_close(
            loaded.feature_extractor.model.state_dict()[name],
            tensor,
            rtol=0,
            atol=0,
        )
    assert untrained_backbone == [True, False]


@pytest.mark.usefixtures("untrained_backbone")
def test_saving_unfitted_model_is_rejected(tmp_path):
    model = PatchCoreModel(input_size=32)
    with pytest.raises(ValueError, match="without a fitted memory bank"):
        model.save(tmp_path / "unfitted.npz")
