from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from visionguard.data import (
    EXPECTED_VISA_CATEGORIES,
    VISA_DATASET_URL,
    VISA_REPOSITORY_URL,
)

REPOSITORY_REVISION = "2a692ab575001cbde74d402d897a7286086c6199"
EXPECTED_COLUMNS = {"object", "split", "label", "image", "mask"}
SPLITS = {"train", "val", "test"}


def _safe_source_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe dataset path in split CSV: {relative}")
    source = root.joinpath(*path.parts)
    if not source.is_file():
        raise FileNotFoundError(f"Split CSV references missing file: {source}")
    return source


def prepare_dataset(
    raw_root: Path,
    split_csv: Path,
    output_root: Path,
    categories: list[str] | None = None,
    archive_metadata_path: Path | None = None,
) -> dict:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw VisA root does not exist: {raw_root}")
    if not split_csv.is_file():
        raise FileNotFoundError(f"Official split CSV not found: {split_csv}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty prepared dataset directory: {output_root}"
        )

    with split_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not EXPECTED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                f"Split CSV must contain columns {sorted(EXPECTED_COLUMNS)}; "
                f"found {reader.fieldnames}"
            )
        rows = list(reader)
    selected_categories = set(categories or EXPECTED_VISA_CATEGORIES)
    invalid_categories = selected_categories - set(EXPECTED_VISA_CATEGORIES)
    if invalid_categories:
        raise ValueError(f"Unsupported VisA categories: {sorted(invalid_categories)}")
    rows = [row for row in rows if row["object"] in selected_categories]
    if not rows:
        raise ValueError("The selected categories have no rows in the official split CSV")

    counts: dict[str, Counter] = {
        category: Counter() for category in selected_categories
    }
    for row in rows:
        category = row["object"]
        split = row["split"]
        label = row["label"]
        if split not in SPLITS:
            raise ValueError(f"Unexpected split '{split}' for {category}")
        if label not in {"normal", "anomaly"}:
            raise ValueError(f"Unexpected label '{label}' for {category}")
        if label == "anomaly" and not row["mask"]:
            raise ValueError(f"Anomaly row has no mask path: {row['image']}")
        if label == "normal" and row["mask"]:
            raise ValueError(f"Normal row unexpectedly has a mask path: {row['image']}")
        source_image = _safe_source_path(raw_root, row["image"])
        try:
            with Image.open(source_image) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"Invalid image referenced by split CSV: {source_image}") from exc
        if row["mask"]:
            source_mask = _safe_source_path(raw_root, row["mask"])
            try:
                with Image.open(source_mask) as mask:
                    mask.verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                raise ValueError(f"Invalid mask referenced by split CSV: {source_mask}") from exc
        counts[category][f"{split}_{label}"] += 1

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = output_root.parent
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}-staging-", dir=staging_parent
    ) as temporary:
        stage = Path(temporary) / output_root.name
        stage.mkdir()
        for row in rows:
            category, split, label = row["object"], row["split"], row["label"]
            category_label = "good" if label == "normal" else "bad"
            image_name = PurePosixPath(row["image"]).name
            source_image = _safe_source_path(raw_root, row["image"])
            destination_image = (
                stage
                / category
                / split
                / category_label
                / image_name
            )
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination_image)

            if label == "anomaly":
                source_mask = _safe_source_path(raw_root, row["mask"])
                mask_name = f"{PurePosixPath(row['image']).stem}_mask.png"
                destination_mask = (
                    stage / category / "ground_truth" / "bad" / mask_name
                )
                destination_mask.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source_mask) as source:
                    mask_array = source.convert("L")
                    binary = mask_array.point(lambda value: 255 if value else 0)
                    binary.save(destination_mask)

        if output_root.exists() and not any(output_root.iterdir()):
            output_root.rmdir()
        resolved_output = output_root.resolve()
        if resolved_output.exists() and not any(resolved_output.iterdir()):
            resolved_output.rmdir()
        shutil.move(str(stage), str(resolved_output))

    category_records = []
    for category in sorted(selected_categories):
        category_counts = counts[category]
        category_records.append(
            {
                "category": category,
                "train_normal": category_counts["train_normal"],
                "val_normal": category_counts["val_normal"],
                "val_anomaly": category_counts["val_anomaly"],
                "test_normal": category_counts["test_normal"],
                "test_anomaly": category_counts["test_anomaly"],
                "test_masks": category_counts["test_anomaly"],
            }
        )
    archive_metadata = None
    if archive_metadata_path is not None and archive_metadata_path.is_file():
        archive_metadata = json.loads(
            archive_metadata_path.read_text(encoding="utf-8")
        )
    return {
        "source_url": VISA_DATASET_URL,
        "repository_url": VISA_REPOSITORY_URL,
        "repository_revision": REPOSITORY_REVISION,
        "license": {
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "license_file": "data/downloads/LICENSE-DATASET",
            "attribution_required": True,
        },
        "archive_integrity": archive_metadata,
        "split_csv": str(split_csv),
        "split_type": "1cls",
        "selected_categories": sorted(selected_categories),
        "rows_prepared": len(rows),
        "prepared_root": str(output_root.resolve()),
        "preparation_date_utc": datetime.now(timezone.utc).isoformat(),
        "preparation": (
            "Equivalent 1cls mapping from the official prepare_data.py: normal "
            "to split/{good}; anomaly to split/bad; test anomaly masks converted "
            "to binary 0/255 under ground_truth/bad. Official CSV train/test "
            "splits are preserved; validation is empty because 1cls.csv defines "
            "no validation split."
        ),
        "categories": category_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare VisA using Amazon's official 1cls split CSV."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/visa_raw"))
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("data/downloads/spot-diff-1cls.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/visa_prepared/1cls")
    )
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument(
        "--archive-metadata",
        type=Path,
        default=Path("reports/visa_archive_integrity.json"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("reports/dataset_manifest.json"))
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.raw_root,
        args.split_csv,
        args.output_root,
        categories=args.categories,
        archive_metadata_path=args.archive_metadata,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
