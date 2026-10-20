from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights

from .config import ModelConfig
from .features import FeatureExtractor, preprocess_image, resize_anomaly_map
from .memory_bank import MemoryBank


class PatchCoreModel:
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        input_size: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        random_seed: Optional[int] = None,
        device: Optional[str] = None,
        pretrained: bool = True,
    ):
        self.config = config or ModelConfig()
        self.input_size = self.config.image_size if input_size is None else int(input_size)
        self.sample_ratio = self.config.sample_ratio if sample_ratio is None else float(sample_ratio)
        self.random_seed = self.config.random_seed if random_seed is None else int(random_seed)
        self.device = self.config.device if device is None else device
        if self.input_size < 1:
            raise ValueError("input_size must be positive")
        if not 0 < self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be in the interval (0, 1]")
        self.feature_extractor = FeatureExtractor(
            device=self.device,
            model_name=self.config.model_name,
            pretrained=pretrained,
        )
        self.memory_bank: Optional[np.ndarray] = None
        self.threshold: Optional[float] = self.config.threshold
        self.pixel_threshold: Optional[float] = None

    @staticmethod
    def _read_image(path: str | Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def fit(self, image_paths: Iterable[str | Path]) -> None:
        feature_vectors = []
        for path in image_paths:
            image = self._read_image(path)
            tensor = preprocess_image(image, size=self.input_size)
            features, _ = self.feature_extractor.extract(tensor)
            feature_vectors.append(features)
        if not feature_vectors:
            raise ValueError("No normal training images were provided to fit the PatchCore model")
        all_features = np.concatenate(feature_vectors, axis=0).astype(np.float32)
        if all_features.shape[0] == 0:
            raise ValueError("No patch features were extracted from the normal training images")
        self.memory_bank = MemoryBank(all_features, sample_ratio=self.sample_ratio, random_seed=self.random_seed).coreset_subsample()

    def _nearest_patch_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank has not been initialized")
        return MemoryBank(self.memory_bank, sample_ratio=1.0).nearest_distances(feature_vectors)

    def _aggregate_image_score(self, patch_distances: np.ndarray) -> float:
        if patch_distances.size == 0:
            return 0.0
        if self.config.image_score_aggregation == "max":
            return float(np.max(patch_distances))
        if self.config.image_score_aggregation == "topk_mean":
            count = max(
                1,
                int(
                    np.ceil(
                        patch_distances.size
                        * self.config.image_score_top_k_fraction
                    )
                ),
            )
            top_values = np.partition(patch_distances, -count)[-count:]
            return float(np.mean(top_values, dtype=np.float64))
        raise ValueError(
            f"Unsupported image score aggregation: "
            f"{self.config.image_score_aggregation}"
        )

    def predict_patch_map(
        self, image: Image.Image | np.ndarray | str | Path
    ) -> Tuple[float, np.ndarray, Tuple[int, int]]:
        if isinstance(image, (str, Path)):
            image = self._read_image(image)
        original_size = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
        tensor = preprocess_image(image, size=self.input_size)
        features, spatial_shape = self.feature_extractor.extract(tensor)
        if self.memory_bank is None:
            raise ValueError("Model has not been fit yet")
        distances = self._nearest_patch_distances(features)
        score = self._aggregate_image_score(distances)
        expected_count = spatial_shape[0] * spatial_shape[1]
        if distances.size != expected_count:
            raise ValueError(
                f"Extracted {distances.size} patch scores for feature map shape {spatial_shape}"
            )
        patch_map = distances.reshape(spatial_shape, order="C")
        return score, patch_map, (original_size[1], original_size[0])

    def predict_image(self, image: Image.Image | np.ndarray | str | Path) -> Tuple[float, np.ndarray]:
        score, patch_map, original_shape = self.predict_patch_map(image)
        anomaly_map = resize_anomaly_map(
            patch_map,
            original_shape,
            interpolation=self.config.anomaly_map_upsampling,
        )
        return score, anomaly_map

    def predict(self, image: Image.Image | np.ndarray | str | Path) -> dict:
        score, anomaly_map = self.predict_image(image)
        if self.threshold is None:
            status = "unclassified"
        else:
            status = "anomalous" if score >= self.threshold else "normal"
        return {"image_score": score, "status": status, "anomaly_map": anomaly_map, "threshold": self.threshold}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "input_size": self.input_size,
            "sample_ratio": self.sample_ratio,
            "random_seed": self.random_seed,
            "threshold": self.threshold,
            "pixel_threshold": self.pixel_threshold,
        }
        if self.memory_bank is None:
            raise ValueError("Cannot save a model without a fitted memory bank")
        backbone_state = self.feature_extractor.model.state_dict()
        payload["backbone_weights"] = (
            str(Wide_ResNet50_2_Weights.DEFAULT)
            if self.config.model_name == "wide_resnet50_2"
            else "untrained"
        )
        arrays = {
            f"backbone.{name}": tensor.detach().cpu().numpy()
            for name, tensor in backbone_state.items()
        }
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(dir=p.parent, suffix=".npz", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary_path,
                metadata=np.asarray(json.dumps(payload)),
                memory_bank=self.memory_bank,
                **arrays,
            )
            os.replace(temporary_path, p)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PatchCoreModel":
        with np.load(path, allow_pickle=False) as artifact:
            payload = json.loads(str(artifact["metadata"].item()))
            memory_bank = np.asarray(artifact["memory_bank"], dtype=np.float32)
            backbone_state = {
                key.removeprefix("backbone."): torch.from_numpy(np.array(artifact[key]))
                for key in artifact.files
                if key.startswith("backbone.")
            }
        MemoryBank(memory_bank, sample_ratio=1.0)
        if not backbone_state:
            raise ValueError("Model artifact is missing the backbone state dictionary")
        config = ModelConfig(**payload["config"])
        model = cls(
            config=config,
            input_size=payload["input_size"],
            sample_ratio=payload["sample_ratio"],
            random_seed=payload["random_seed"],
            device=device,
            pretrained=False,
        )
        model.feature_extractor.model.load_state_dict(backbone_state, strict=True)
        model.memory_bank = memory_bank
        model.threshold = payload.get("threshold")
        model.pixel_threshold = payload.get("pixel_threshold")
        return model
