from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class ModelConfig:
    dataset_root: str = "data/mvtec_ad"
    category: str = "bottle"
    image_size: int = 224
    batch_size: int = 8
    sample_ratio: float = 0.01
    random_seed: int = 42
    device: str = "cpu"
    model_name: str = "wide_resnet50_2"
    image_score_aggregation: str = "max"
    image_score_top_k_fraction: float = 0.01
    anomaly_map_upsampling: str = "bilinear"
    threshold: Optional[float] = None
    heatmap_dir: str = "artifacts/heatmaps"
    model_path: str = "artifacts/model.npz"
    max_upload_bytes: int = 10 * 1024 * 1024
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "visionguard"

    def __post_init__(self) -> None:
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 < self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be in the interval (0, 1]")
        if self.image_score_aggregation not in {"max", "topk_mean"}:
            raise ValueError(
                "image_score_aggregation must be 'max' or 'topk_mean'"
            )
        if not 0 < self.image_score_top_k_fraction <= 1:
            raise ValueError("image_score_top_k_fraction must be in the interval (0, 1]")
        if self.anomaly_map_upsampling not in {"bilinear", "nearest"}:
            raise ValueError(
                "anomaly_map_upsampling must be 'bilinear' or 'nearest'"
            )
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be positive")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")

    def to_dict(self) -> dict:
        return {
            "dataset_root": self.dataset_root,
            "category": self.category,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "sample_ratio": self.sample_ratio,
            "random_seed": self.random_seed,
            "device": self.device,
            "model_name": self.model_name,
            "image_score_aggregation": self.image_score_aggregation,
            "image_score_top_k_fraction": self.image_score_top_k_fraction,
            "anomaly_map_upsampling": self.anomaly_map_upsampling,
            "threshold": self.threshold,
            "heatmap_dir": self.heatmap_dir,
            "model_path": self.model_path,
            "max_upload_bytes": self.max_upload_bytes,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "mlflow_experiment": self.mlflow_experiment,
        }

def load_config_from_env() -> ModelConfig:
    import os

    cfg = ModelConfig()
    cfg.dataset_root = os.getenv("VISIONGUARD_DATASET_ROOT", cfg.dataset_root)
    cfg.category = os.getenv("VISIONGUARD_CATEGORY", cfg.category)
    cfg.image_size = int(os.getenv("VISIONGUARD_IMAGE_SIZE", str(cfg.image_size)))
    cfg.batch_size = int(os.getenv("VISIONGUARD_BATCH_SIZE", str(cfg.batch_size)))
    cfg.sample_ratio = float(os.getenv("VISIONGUARD_SAMPLE_RATIO", str(cfg.sample_ratio)))
    cfg.random_seed = int(os.getenv("VISIONGUARD_RANDOM_SEED", str(cfg.random_seed)))
    cfg.device = os.getenv("VISIONGUARD_DEVICE", cfg.device)
    cfg.model_name = os.getenv("VISIONGUARD_MODEL_NAME", cfg.model_name)
    cfg.image_score_aggregation = os.getenv(
        "VISIONGUARD_IMAGE_SCORE_AGGREGATION", cfg.image_score_aggregation
    )
    cfg.image_score_top_k_fraction = float(
        os.getenv(
            "VISIONGUARD_IMAGE_SCORE_TOP_K_FRACTION",
            str(cfg.image_score_top_k_fraction),
        )
    )
    cfg.anomaly_map_upsampling = os.getenv(
        "VISIONGUARD_ANOMALY_MAP_UPSAMPLING",
        cfg.anomaly_map_upsampling,
    )
    cfg.threshold = os.getenv("VISIONGUARD_THRESHOLD")
    if cfg.threshold is not None:
        cfg.threshold = float(cfg.threshold)
    cfg.heatmap_dir = os.getenv("VISIONGUARD_HEATMAP_DIR", cfg.heatmap_dir)
    cfg.model_path = os.getenv("VISIONGUARD_MODEL_PATH", cfg.model_path)
    cfg.max_upload_bytes = int(os.getenv("VISIONGUARD_MAX_UPLOAD_BYTES", str(cfg.max_upload_bytes)))
    cfg.mlflow_tracking_uri = os.getenv("VISIONGUARD_MLFLOW_TRACKING_URI")
    cfg.mlflow_experiment = os.getenv("VISIONGUARD_MLFLOW_EXPERIMENT", cfg.mlflow_experiment)
    return cfg
