from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

EXPECTED_MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
MVTEC_DATASET_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
EXPECTED_VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]
VISA_DATASET_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_REPOSITORY_URL = "https://github.com/amazon-science/spot-diff"


@dataclass
class DatasetItem:
    image_path: Path
    mask_path: Optional[Path]
    label: str
    category: str

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

    @property
    def mask(self) -> Optional[np.ndarray]:
        if self.mask_path is None:
            return None
        with Image.open(self.mask_path) as mask:
            return np.asarray(mask).copy()


def _is_valid_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}


def list_categories(dataset_root: str | Path) -> List[str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    categories = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "train").exists() and (child / "test").exists():
            categories.append(child.name)
    return categories


def validate_dataset(dataset_root: str | Path, category: Optional[str] = None) -> List[str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    selected = list_categories(root) if category is None else [category]
    if category is not None and category not in selected:
        raise ValueError(f"Requested category '{category}' not found in '{root}'")

    validated = []
    for cat in selected:
        category_root = root / cat
        train_root = category_root / "train"
        test_root = category_root / "test"
        if not train_root.exists() or not test_root.exists():
            raise FileNotFoundError(f"Missing train/test folders in dataset category '{cat}'")

        train_good = sorted((train_root / "good").glob("*"))
        if not train_good:
            raise ValueError(f"No normal training images found for category '{cat}'")
        bad_files = [p for p in train_good if not _is_valid_file(p)]
        if bad_files:
            raise ValueError(f"Found malformed training image files in '{cat}': {bad_files[:3]}")

        test_dirs = [p for p in test_root.iterdir() if p.is_dir()]
        if not test_dirs:
            raise ValueError(f"No test split folders found for category '{cat}'")

        for label_dir in test_dirs:
            files = [p for p in label_dir.iterdir() if p.is_file() and _is_valid_file(p)]
            if not files:
                raise ValueError(f"No valid test images found under '{label_dir}' for category '{cat}'")
        validated.append(cat)
    return validated


def load_train_images(dataset_root: str | Path, category: str) -> List[Path]:
    root = Path(dataset_root) / category / "train" / "good"
    if not root.exists():
        raise FileNotFoundError(f"Training split not found for '{category}' at '{root}'")
    return sorted([p for p in root.iterdir() if _is_valid_file(p)])


def load_test_items(dataset_root: str | Path, category: str) -> List[DatasetItem]:
    root = Path(dataset_root) / category / "test"
    items: List[DatasetItem] = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        defect_type = label_dir.name
        for image_path in sorted(label_dir.iterdir()):
            if not _is_valid_file(image_path):
                continue
            mask_path = None
            if defect_type != "good":
                masks_root = Path(dataset_root) / category / "ground_truth" / defect_type
                candidate = masks_root / f"{image_path.stem}_mask.png"
                if candidate.exists():
                    mask_path = candidate
            items.append(
                DatasetItem(
                    image_path=image_path,
                    mask_path=mask_path,
                    label="good" if defect_type == "good" else defect_type,
                    category=category,
                )
            )
    return items


def build_synthetic_mvtec_dataset(root: str | Path, category: str = "tile", train_good: int = 8, test_good: int = 3, test_bad: int = 3, image_size: int = 64) -> Path:
    root = Path(root)
    category_root = root / category
    train_root = category_root / "train" / "good"
    test_root = category_root / "test" / "good"
    defective_root = category_root / "test" / "synthetic_defect"
    gt_root = category_root / "ground_truth" / "synthetic_defect"
    for path in [train_root, test_root, defective_root, gt_root]:
        path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for i in range(train_good):
        arr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        arr[:, :, 0] = 140 + rng.integers(0, 40, size=(image_size, image_size))
        arr[:, :, 1] = 150 + rng.integers(0, 40, size=(image_size, image_size))
        arr[:, :, 2] = 160 + rng.integers(0, 40, size=(image_size, image_size))
        arr[10:20, 10:20] = 200
        Image.fromarray(arr, "RGB").save(train_root / f"train_{i:02d}.png")

    for i in range(test_good):
        arr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        arr[:, :, 0] = 120 + rng.integers(0, 30, size=(image_size, image_size))
        arr[:, :, 1] = 130 + rng.integers(0, 30, size=(image_size, image_size))
        arr[:, :, 2] = 140 + rng.integers(0, 30, size=(image_size, image_size))
        Image.fromarray(arr, "RGB").save(test_root / f"good_{i:02d}.png")

    for i in range(test_bad):
        arr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        arr[:, :, 0] = 120 + rng.integers(0, 20, size=(image_size, image_size))
        arr[:, :, 1] = 130 + rng.integers(0, 20, size=(image_size, image_size))
        arr[:, :, 2] = 140 + rng.integers(0, 20, size=(image_size, image_size))
        anomaly = np.zeros((image_size, image_size), dtype=np.uint8)
        anomaly[20:40, 20:45] = 255
        arr[20:40, 20:45] = 255
        Image.fromarray(arr, "RGB").save(defective_root / f"defective_{i:02d}.png")
        Image.fromarray(anomaly, "L").save(gt_root / f"defective_{i:02d}_mask.png")

    return category_root


def save_dataset_manifest(dataset_root: str | Path, manifest_path: str | Path) -> None:
    root = Path(dataset_root)
    manifest = {"categories": []}
    for category in list_categories(root):
        manifest["categories"].append({
            "name": category,
            "train_count": len(load_train_images(root, category)),
            "test_count": len(load_test_items(root, category)),
        })
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
