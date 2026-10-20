from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from visionguard.data import (
    EXPECTED_MVTEC_CATEGORIES,
    MVTEC_DATASET_URL,
    _is_valid_file,
    list_categories,
    load_test_items,
    load_train_images,
)


def build_manifest(dataset_root: str | Path, category: str | None = None) -> dict:
    root = Path(dataset_root)
    if not root.is_dir():
        return {
            "source": MVTEC_DATASET_URL,
            "download_status": "not downloaded",
            "terms_status": "not verified; official site unreachable",
            "dataset_root": str(root),
            "validation_date_utc": datetime.now(timezone.utc).isoformat(),
            "download_date_utc": None,
            "categories": [],
            "missing_categories": EXPECTED_MVTEC_CATEGORIES.copy(),
            "image_counts": {},
            "missing_masks": [],
            "malformed_images": [],
            "validation_errors": [f"Dataset directory not found: {root}"],
            "valid": False,
        }

    discovered_all = list_categories(root)
    discovered = (
        [category]
        if category in discovered_all
        else []
    ) if category else discovered_all
    missing_categories = sorted(set(EXPECTED_MVTEC_CATEGORIES) - set(discovered_all))
    errors = []
    if category is not None and category not in discovered_all:
        errors.append(f"Requested category '{category}' is missing from {root}")
    malformed_images = []
    missing_masks = []
    category_records = []
    image_counts = {}

    for category in discovered:
        category_path = root / category
        try:
            train_images = load_train_images(root, category)
            test_items = load_test_items(root, category)
        except (OSError, ValueError) as exc:
            errors.append(f"{category}: {exc}")
            continue

        for path in train_images + [item.image_path for item in test_items]:
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                malformed_images.append({"path": str(path), "error": str(exc)})

        defect_counts: dict[str, int] = {}
        for item in test_items:
            if item.label != "good":
                defect_counts[item.label] = defect_counts.get(item.label, 0) + 1
                if item.mask_path is None or not _is_valid_file(item.mask_path):
                    missing_masks.append(
                        {"image": str(item.image_path), "expected_mask": str(
                            category_path / "ground_truth" / item.label / f"{item.image_path.stem}_mask.png"
                        )}
                    )
                else:
                    try:
                        with Image.open(item.mask_path) as mask:
                            mask.verify()
                    except (OSError, UnidentifiedImageError, ValueError) as exc:
                        malformed_images.append({"path": str(item.mask_path), "error": str(exc)})

        test_counts = {"good": sum(item.label == "good" for item in test_items), **defect_counts}
        counts = {
            "train_good": len(train_images),
            "test_good": test_counts.pop("good", 0),
            "test_anomalous": sum(test_counts.values()),
            "test_by_defect": test_counts,
            "masks_found": sum(
                item.mask_path is not None
                for item in test_items
                if item.label != "good"
            ),
        }
        image_counts[category] = counts
        category_records.append({"category": category, **counts})
        if not train_images:
            errors.append(f"{category}: train/good contains no readable image files")
        if not test_items:
            errors.append(f"{category}: test contains no valid image files")

    if category is None and missing_categories:
        errors.append(f"Missing expected MVTec categories: {', '.join(missing_categories)}")
    if missing_masks:
        errors.append(f"{len(missing_masks)} anomalous test image(s) are missing masks")
    if malformed_images:
        errors.append(f"{len(malformed_images)} image or mask file(s) are unreadable")

    return {
        "source": MVTEC_DATASET_URL,
        "download_status": "present; validation attempted",
        "terms_status": "not verified; official site unreachable",
        "dataset_root": str(root.resolve()),
        "validation_date_utc": datetime.now(timezone.utc).isoformat(),
        "download_date_utc": None,
        "categories": category_records,
        "discovered_category_names": discovered,
        "scope": category or "all categories",
        "missing_categories": missing_categories,
        "image_counts": image_counts,
        "missing_masks": missing_masks,
        "malformed_images": malformed_images,
        "validation_errors": errors,
        "valid": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an extracted MVTec AD directory.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_manifest.json"))
    parser.add_argument("--category", help="Validate one category without requiring all 15.")
    args = parser.parse_args()

    manifest = build_manifest(args.dataset_root, category=args.category)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
