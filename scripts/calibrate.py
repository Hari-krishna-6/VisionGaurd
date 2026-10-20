from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
import torchvision

from visionguard.config import ModelConfig, load_config_from_env
from visionguard.data import (
    build_synthetic_mvtec_dataset,
    list_categories,
    load_train_images,
    validate_dataset,
)
from visionguard.evaluation import determine_pixel_threshold, determine_threshold
from visionguard.mlflow_utils import log_experiment
from visionguard.model import PatchCoreModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibrate(config: ModelConfig, dataset_root: Path, dataset_type: str) -> None:
    validate_dataset(dataset_root, config.category)
    artifact_path = Path(config.model_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{artifact_path}'. Run 'python -m scripts.train' first."
        )

    model = PatchCoreModel.load(artifact_path, device=config.device)
    train_images = load_train_images(dataset_root, config.category)
    started = perf_counter()
    model.threshold = determine_threshold(train_images, model, quantile=0.99)
    model.pixel_threshold = determine_pixel_threshold(
        train_images,
        model,
        quantile=0.995,
    )
    calibration_seconds = perf_counter() - started
    model.save(artifact_path)

    metadata = {
        "dataset_root": str(dataset_root.resolve()),
        "dataset_type": dataset_type,
        "category": config.category,
        "calibration_split": "train/good (normal images only)",
        "image_threshold_quantile": 0.99,
        "image_threshold": model.threshold,
        "pixel_threshold_quantile": 0.995,
        "pixel_threshold": model.pixel_threshold,
        "normal_images_used": len(train_images),
        "random_seed": model.random_seed,
        "image_size": model.input_size,
        "memory_bank_size": int(model.memory_bank.shape[0]),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "artifact_sha256": _sha256(artifact_path),
    }
    metadata_path = artifact_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "calibration_seconds": calibration_seconds}, indent=2))
    log_experiment(
        run_name=f"{config.category}-calibration",
        params={
            **config.to_dict(),
            "sample_ratio": model.sample_ratio,
            "random_seed": model.random_seed,
            "image_size": model.input_size,
            "dataset_type": dataset_type,
            "dataset_identity": str(dataset_root.resolve()),
            "calibration_split": "train/good",
            "normal_images_used": len(train_images),
            "image_quantile": 0.99,
            "pixel_quantile": 0.995,
        },
        metrics={
            "image_threshold": model.threshold,
            "pixel_threshold": model.pixel_threshold,
            "calibration_seconds": calibration_seconds,
        },
        artifact_paths=(artifact_path, metadata_path),
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.mlflow_experiment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate thresholds using normal train/good images only."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional dataset root override; defaults to VISIONGUARD_DATASET_ROOT.",
    )
    parser.add_argument("--category", help="Category to calibrate.")
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Calibrate each discovered category model.",
    )
    parser.add_argument("--model", type=Path, help="Optional model artifact path override.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a temporary synthetic dataset for code-path validation only.",
    )
    args = parser.parse_args()
    if args.category and args.all_categories:
        parser.error("--category and --all-categories cannot be combined")

    config = load_config_from_env()
    if args.model is not None:
        config.model_path = str(args.model)
    dataset_root = args.dataset_root or Path(config.dataset_root)
    if args.dataset_root is not None:
        dataset_type = (
            "VisA"
            if "visa" in str(dataset_root).replace("\\", "/").lower()
            else "MVTec AD"
        )
    elif args.smoke_test:
        with TemporaryDirectory(prefix="visionguard-smoke-") as temporary_root:
            dataset_root = Path(temporary_root) / "dataset"
            build_synthetic_mvtec_dataset(dataset_root, category=config.category)
            _calibrate(config, dataset_root, "synthetic smoke test (not MVTec AD)")
            return
    else:
        dataset_type = (
            "VisA"
            if "visa" in str(dataset_root).replace("\\", "/").lower()
            else "MVTec AD"
        )

    categories = (
        list_categories(dataset_root)
        if args.all_categories
        else [args.category or config.category]
    )
    for category in categories:
        category_config = replace(config, category=category)
        if args.all_categories:
            category_config.model_path = str(
                Path(config.model_path).parent
                / category
                / Path(config.model_path).name
            )
        _calibrate(category_config, dataset_root, dataset_type)


if __name__ == "__main__":
    main()
