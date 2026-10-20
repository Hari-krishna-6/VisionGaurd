from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import ModelConfig
from .features import _load_backbone, preprocess_image, resize_anomaly_map
from .memory_bank import MemoryBank

PROJECTED_CHANNELS = 128
SCALE_COUNT = 3
CORESET_CANDIDATE_LIMIT = 6_000


def combine_multiscale_maps(
    layer1: torch.Tensor,
    layer2: torch.Tensor,
    layer3: torch.Tensor,
    projected_channels: int = PROJECTED_CHANNELS,
    projections: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Project, align, and equally weight layer1/layer2/layer3 descriptors."""
    maps = (layer1, layer2, layer3)
    if any(value.ndim != 4 or value.shape[0] != 1 for value in maps):
        raise ValueError("Expected three single-image NCHW feature maps")
    if any(value.device != layer1.device for value in maps):
        raise ValueError("All feature maps must be on the same device")
    if projected_channels < 1:
        raise ValueError("projected_channels must be positive")

    output_shape = layer1.shape[-2:]
    if projections is None:
        generator = torch.Generator(device="cpu").manual_seed(42)
        projections = tuple(
            torch.randn(
                (value.shape[1], projected_channels),
                generator=generator,
                dtype=torch.float32,
            )
            .div_(math.sqrt(projected_channels))
            .to(device=value.device)
            for value in maps
        )
    if len(projections) != SCALE_COUNT or any(
        projection.shape != (value.shape[1], projected_channels)
        for value, projection in zip(maps, projections)
    ):
        raise ValueError("Projection matrices must match each feature-map channel count")

    projected_maps = []
    for feature_map, projection in zip(maps, projections):
        if feature_map.shape[-2:] != output_shape:
            feature_map = F.interpolate(
                feature_map,
                size=output_shape,
                mode="bilinear",
                align_corners=False,
            )
        channel_last = feature_map.permute(0, 2, 3, 1)
        projected = torch.matmul(channel_last, projection)
        projected = F.normalize(projected, p=2, dim=-1, eps=1e-12)
        projected_maps.append(projected / math.sqrt(SCALE_COUNT))

    combined = torch.cat(projected_maps, dim=-1)
    return combined[0].reshape(
        output_shape[0] * output_shape[1],
        projected_channels * SCALE_COUNT,
    )


class MultiscaleFeatureExtractor:
    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        random_seed: int = 42,
        projected_channels: int = PROJECTED_CHANNELS,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.projected_channels = projected_channels
        self.model = _load_backbone(model_name, pretrained=pretrained).to(self.device)
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        self.projections = tuple(
            torch.randn(
                (channels, projected_channels),
                generator=generator,
                dtype=torch.float32,
            )
            .div_(math.sqrt(projected_channels))
            .to(device=self.device)
            for channels in (256, 512, 1024)
        )
        self.last_layer_shapes: dict[str, tuple[int, ...]] = {}

    def extract(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int]]:
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError(
                "MultiscaleFeatureExtractor.extract expects a single NCHW image tensor"
            )
        with torch.inference_mode():
            x = image_tensor.to(self.device)
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            layer1 = self.model.layer1(x)
            layer2 = self.model.layer2(layer1)
            layer3 = self.model.layer3(layer2)
            self.last_layer_shapes = {
                "layer1": tuple(layer1.shape),
                "layer2": tuple(layer2.shape),
                "layer3": tuple(layer3.shape),
            }
            vectors = combine_multiscale_maps(
                layer1,
                layer2,
                layer3,
                projected_channels=self.projected_channels,
                projections=self.projections,
            )
            height, width = layer1.shape[-2:]
            output = vectors.cpu().numpy().astype(np.float32, copy=False)
        return output, (height, width)


class MultiscalePatchCoreModel:
    """Experiment-only multiscale PatchCore model; production model is untouched."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        projected_channels: int = PROJECTED_CHANNELS,
        pretrained: bool = True,
    ):
        self.config = config or ModelConfig()
        self.input_size = self.config.image_size
        self.sample_ratio = self.config.sample_ratio
        self.random_seed = self.config.random_seed
        self.feature_extractor = MultiscaleFeatureExtractor(
            device=self.config.device,
            model_name=self.config.model_name,
            pretrained=pretrained,
            random_seed=self.random_seed,
            projected_channels=projected_channels,
        )
        self.projected_channels = projected_channels
        self.memory_bank: Optional[np.ndarray] = None
        self.threshold: Optional[float] = self.config.threshold
        self.pixel_threshold: Optional[float] = None
        self.fit_metadata: dict[str, object] = {}

    def fit(self, image_paths: Iterable[str | Path]) -> None:
        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ValueError("No normal training images were provided")

        patches_per_image = (self.input_size // 4) ** 2
        total_patches = len(paths) * patches_per_image
        target_count = max(1, int(math.ceil(total_patches * self.sample_ratio)))
        candidate_count = min(
            total_patches,
            max(target_count, CORESET_CANDIDATE_LIMIT),
        )
        selected_indices = np.sort(
            np.random.default_rng(self.random_seed).choice(
                total_patches,
                size=candidate_count,
                replace=False,
            )
        )
        candidates = np.empty(
            (candidate_count, self.projected_channels * SCALE_COUNT),
            dtype=np.float32,
        )
        selected_cursor = 0

        for image_index, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            vectors, shape = self.feature_extractor.extract(
                preprocess_image(image, size=self.input_size)
            )
            if shape[0] * shape[1] != patches_per_image:
                raise ValueError(
                    f"Unexpected feature grid {shape}; expected "
                    f"{self.input_size // 4}x{self.input_size // 4}"
                )
            if vectors.shape != (
                patches_per_image,
                self.projected_channels * SCALE_COUNT,
            ):
                raise ValueError(f"Unexpected multiscale descriptor shape: {vectors.shape}")

            image_start = image_index * patches_per_image
            image_stop = image_start + patches_per_image
            next_cursor = int(np.searchsorted(selected_indices, image_stop, side="left"))
            if next_cursor > selected_cursor:
                local_indices = (
                    selected_indices[selected_cursor:next_cursor] - image_start
                )
                candidates[selected_cursor:next_cursor] = vectors[local_indices]
            selected_cursor = next_cursor
            if (image_index + 1) % 100 == 0:
                print(
                    f"fit progress: {image_index + 1}/{len(paths)} normal images",
                    flush=True,
                )

        if selected_cursor != candidate_count:
            raise RuntimeError(
                f"Selected {selected_cursor} candidate patches; expected {candidate_count}"
            )
        bank = MemoryBank(
            candidates,
            sample_ratio=target_count / candidate_count,
            random_seed=self.random_seed,
        )
        self.memory_bank = bank.coreset_subsample()
        self.fit_metadata = {
            "normal_training_image_count": len(paths),
            "total_training_patch_count": total_patches,
            "random_candidate_count": candidate_count,
            "memory_bank_patch_count": int(self.memory_bank.shape[0]),
            "memory_bank_dimension": int(self.memory_bank.shape[1]),
            "random_seed": self.random_seed,
            "coreset_selection": "greedy farthest-point selection over seeded uniform candidate pool",
        }

    def _nearest_patch_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank has not been initialized")
        return MemoryBank(self.memory_bank, sample_ratio=1.0).nearest_distances(
            feature_vectors,
            query_chunk_size=128,
            bank_chunk_size=1024,
        )

    def predict_patch_map(
        self, image: Image.Image | np.ndarray | str | Path
    ) -> Tuple[float, np.ndarray, Tuple[int, int]]:
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        original_size = (
            image.size
            if isinstance(image, Image.Image)
            else (image.shape[1], image.shape[0])
        )
        features, spatial_shape = self.feature_extractor.extract(
            preprocess_image(image, size=self.input_size)
        )
        if self.memory_bank is None:
            raise ValueError("Model has not been fit yet")
        distances = self._nearest_patch_distances(features)
        expected_count = spatial_shape[0] * spatial_shape[1]
        if distances.size != expected_count:
            raise ValueError(
                f"Extracted {distances.size} patch scores for feature map {spatial_shape}"
            )
        patch_map = distances.reshape(spatial_shape, order="C")
        return (
            float(np.max(distances)),
            patch_map,
            (original_size[1], original_size[0]),
        )

    def predict_image(
        self, image: Image.Image | np.ndarray | str | Path
    ) -> Tuple[float, np.ndarray]:
        score, patch_map, source_shape = self.predict_patch_map(image)
        anomaly_map = resize_anomaly_map(
            patch_map,
            source_shape,
            interpolation=self.config.anomaly_map_upsampling,
        )
        return score, anomaly_map

    def save(self, path: str | Path) -> None:
        if self.memory_bank is None:
            raise ValueError("Cannot save a multiscale model without a fitted memory bank")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "config": self.config.to_dict(),
            "input_size": self.input_size,
            "sample_ratio": self.sample_ratio,
            "random_seed": self.random_seed,
            "projected_channels": self.projected_channels,
            "threshold": self.threshold,
            "pixel_threshold": self.pixel_threshold,
            "fit_metadata": self.fit_metadata,
        }
        arrays = {
            f"backbone.{name}": tensor.detach().cpu().numpy()
            for name, tensor in self.feature_extractor.model.state_dict().items()
        }
        arrays.update(
            {
                f"projection.{index}": projection.detach().cpu().numpy()
                for index, projection in enumerate(self.feature_extractor.projections)
            }
        )
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, suffix=".npz", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary_path,
                metadata=np.asarray(json.dumps(metadata)),
                memory_bank=self.memory_bank,
                **arrays,
            )
            os.replace(temporary_path, output)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(
        cls, path: str | Path, device: str = "cpu"
    ) -> "MultiscalePatchCoreModel":
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            memory_bank = np.asarray(artifact["memory_bank"], dtype=np.float32)
            backbone_state = {
                key.removeprefix("backbone."): torch.from_numpy(np.array(artifact[key]))
                for key in artifact.files
                if key.startswith("backbone.")
            }
            projections = tuple(
                torch.from_numpy(np.array(artifact[f"projection.{index}"]))
                for index in range(SCALE_COUNT)
            )

        MemoryBank(memory_bank, sample_ratio=1.0)
        config_values = metadata["config"]
        config_values["device"] = device
        config = ModelConfig(**config_values)
        model = cls(
            config=config,
            projected_channels=int(metadata["projected_channels"]),
            pretrained=False,
        )
        model.feature_extractor.model.load_state_dict(backbone_state, strict=True)
        model.feature_extractor.projections = tuple(
            projection.to(model.feature_extractor.device)
            for projection in projections
        )
        model.memory_bank = memory_bank
        model.threshold = metadata.get("threshold")
        model.pixel_threshold = metadata.get("pixel_threshold")
        model.fit_metadata = metadata.get("fit_metadata", {})
        return model

