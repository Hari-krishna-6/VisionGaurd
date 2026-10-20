from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import models
from torchvision.models import Wide_ResNet50_2_Weights

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ANOMALY_MAP_INTERPOLATIONS = {
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def resize_anomaly_map(
    anomaly_map: np.ndarray,
    output_shape: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Anomaly map must be a finite 2D array")
    height, width = output_shape
    if height < 1 or width < 1:
        raise ValueError("Output dimensions must be positive")
    try:
        interpolation_code = ANOMALY_MAP_INTERPOLATIONS[interpolation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported anomaly-map interpolation: {interpolation}"
        ) from exc
    return cv2.resize(
        values,
        (width, height),
        interpolation=interpolation_code,
    )


def nearest_resize_pixel_counts(
    input_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
) -> np.ndarray:
    input_height, input_width = input_shape
    output_height, output_width = output_shape
    if min(input_height, input_width, output_height, output_width) < 1:
        raise ValueError("Input and output dimensions must be positive")
    patch_count = input_height * input_width
    if patch_count > np.iinfo(np.uint16).max:
        raise ValueError("Input grid is too large for nearest-resize index tracking")
    patch_indices = np.arange(patch_count, dtype=np.uint16).reshape(input_shape)
    resized_indices = cv2.resize(
        patch_indices,
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.bincount(
        resized_indices.reshape(-1),
        minlength=patch_count,
    ).astype(np.int64, copy=False)


def preprocess_image(image: Image.Image | np.ndarray, size: int = 224) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    else:
        img = image.convert("RGB")

    resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN.reshape(1, 1, 3)) / STD.reshape(1, 1, 3)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def _load_backbone(weights_name: str = "wide_resnet50_2", pretrained: bool = True):
    if weights_name == "wide_resnet50_2" and pretrained:
        model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
    else:
        model = models.wide_resnet50_2(weights=None)
    model.eval()
    return model


class FeatureExtractor:
    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "wide_resnet50_2",
        pretrained: bool = True,
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = _load_backbone(model_name, pretrained=pretrained).to(self.device)

    def extract(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, int]]:
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("FeatureExtractor.extract expects a single NCHW image tensor")
        with torch.inference_mode():
            x = image_tensor.to(self.device)
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            layer2 = self.model.layer2(x)
            layer3 = self.model.layer3(layer2)

            # Align the coarser layer3 map to layer2 before concatenating channels.
            if layer2.shape[-2:] != layer3.shape[-2:]:
                layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)

            combined = torch.cat([layer2, layer3], dim=1)
            height, width = combined.shape[-2:]
            output = combined[0].permute(1, 2, 0).reshape(height * width, combined.shape[1]).cpu().numpy()
        return output, (height, width)
