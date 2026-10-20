from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from dataclasses import replace
import importlib.metadata
import platform

from visionguard.config import ModelConfig, load_config_from_env
from visionguard.data import build_synthetic_mvtec_dataset, list_categories, load_train_images, validate_dataset
from visionguard.mlflow_utils import log_experiment
from visionguard.model import PatchCoreModel


def _dataset_type(dataset_root: Path) -> str:
    return (
        "VisA"
        if "visa" in str(dataset_root).replace("\\", "/").lower()
        else "MVTec AD"
    )


def _train(cfg: ModelConfig, dataset_root: Path, dataset_type: str) -> None:
    validate_dataset(dataset_root, cfg.category)

    train_paths = load_train_images(dataset_root, cfg.category)
    model = PatchCoreModel(
        config=cfg,
        input_size=cfg.image_size,
        sample_ratio=cfg.sample_ratio,
        random_seed=cfg.random_seed,
        device=cfg.device,
    )
    started = perf_counter()
    model.fit(train_paths)
    training_seconds = perf_counter() - started
    output = Path(cfg.model_path)
    model.save(output)
    digest = hashlib.sha256()
    with output.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root.resolve()),
                "dataset_type": dataset_type,
                "category": cfg.category,
                "training_split": "train/good",
                "normal_images_used": len(train_paths),
                "backbone": cfg.model_name,
                "pretrained_weights": "TorchVision IMAGENET1K_V2",
                "input_size": cfg.image_size,
                "sample_ratio": cfg.sample_ratio,
                "random_seed": cfg.random_seed,
                "memory_bank_size": int(model.memory_bank.shape[0]),
                "feature_dimensions": int(model.memory_bank.shape[1]),
                "torch_version": __import__("torch").__version__,
                "torchvision_version": __import__("torchvision").__version__,
                "python_version": platform.python_version(),
                "artifact_size_bytes": output.stat().st_size,
                "artifact_sha256": digest.hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = {
        "training_seconds": training_seconds,
        "memory_bank_size": float(model.memory_bank.shape[0]),
    }
    params = {
        **cfg.to_dict(),
        "dataset_type": dataset_type,
        "dataset_identity": str(dataset_root.resolve()),
        "git_commit": "unavailable (workspace has no .git directory)",
        "python_version": platform.python_version(),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "Pillow", "opencv-python", "scikit-learn")
        },
        "training_images": len(train_paths),
        "memory_bank_size": model.memory_bank.shape[0],
        "threshold_calibration": "pending separate normal-training calibration",
    }
    log_experiment(
        run_name=f"{cfg.category}-train",
        params=params,
        metrics=metrics,
        artifact_paths=(output, metadata_path),
        tracking_uri=cfg.mlflow_tracking_uri,
        experiment_name=cfg.mlflow_experiment,
    )
    print(
        f"Training complete ({dataset_type}): "
        f"{len(train_paths)} normal images, memory bank {model.memory_bank.shape[0]} patches, "
        f"artifact {output}; run 'python -m scripts.calibrate' before inference."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a normal-only PatchCore memory bank.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate and train against a tiny synthetic dataset instead of MVTec AD.",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Train a separate model for each discovered dataset category.",
    )
    args = parser.parse_args()

    cfg = load_config_from_env()
    if args.smoke_test:
        with TemporaryDirectory(prefix="visionguard-smoke-") as temporary_root:
            dataset_root = Path(temporary_root) / "dataset"
            build_synthetic_mvtec_dataset(dataset_root, category=cfg.category)
            cfg.dataset_root = str(dataset_root)
            _train(cfg, dataset_root, "synthetic smoke test (not MVTec AD)")
    else:
        dataset_root = Path(cfg.dataset_root)
        dataset_type = _dataset_type(dataset_root)
        categories = (
            list_categories(dataset_root)
            if args.all_categories
            else [cfg.category]
        )
        if not categories:
            raise ValueError(f"No dataset categories found under '{dataset_root}'")
        for category in categories:
            category_config = replace(cfg, category=category)
            if args.all_categories:
                category_config.model_path = str(
                    Path(cfg.model_path).parent / category / Path(cfg.model_path).name
                )
            _train(category_config, dataset_root, dataset_type)


if __name__ == "__main__":
    main()
