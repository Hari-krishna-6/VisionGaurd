from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


def generate_heatmap(anomaly_map: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    heatmap = np.asarray(anomaly_map, dtype=np.float32)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Anomaly map must be a finite 2D array")
    heatmap = cv2.resize(heatmap, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    minimum = float(heatmap.min())
    span = float(heatmap.max()) - minimum
    normalized = np.zeros_like(heatmap) if span <= 1e-12 else (heatmap - minimum) / span
    heatmap_bgr = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def _save_legend(path: Path) -> None:
    gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (24, 1))
    gradient_bgr = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    canvas = np.full((52, 300, 3), 255, dtype=np.uint8)
    canvas[4:28, 22:278] = gradient_bgr
    cv2.putText(canvas, "Low", (22, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(canvas, "High", (244, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(path)


def overlay_heatmap(image: Image.Image | np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if isinstance(image, Image.Image):
        image_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        image_arr = np.asarray(image, dtype=np.uint8)
    heatmap_color = generate_heatmap(anomaly_map, (image_arr.shape[0], image_arr.shape[1]))
    overlay = cv2.addWeighted(image_arr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


def save_heatmap_artifacts(image: Image.Image | np.ndarray, anomaly_map: np.ndarray, output_dir: str | Path, stem: str = "prediction") -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        orig = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        orig = np.asarray(image, dtype=np.uint8)

    original_path = output_dir / f"{stem}_original.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    legend_path = output_dir / f"{stem}_legend.png"

    Image.fromarray(orig).save(original_path)
    heatmap_color = generate_heatmap(anomaly_map, (orig.shape[0], orig.shape[1]))
    Image.fromarray(heatmap_color).save(heatmap_path)
    Image.fromarray(overlay_heatmap(orig, anomaly_map)).save(overlay_path)
    _save_legend(legend_path)
    return {
        "original": str(original_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
        "legend": str(legend_path),
    }
