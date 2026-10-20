from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.extract_visa_archive import extract_archive
from scripts.prepare_visa import prepare_dataset
from visionguard.data import (
    EXPECTED_VISA_CATEGORIES,
    VISA_DATASET_URL,
    load_test_items,
    load_train_images,
    validate_dataset,
)


def _make_raw_fixture(root: Path, split_csv: Path) -> None:
    rows = [
        {
            "object": "candle",
            "split": "train",
            "label": "normal",
            "image": "candle/Data/Images/Normal/train.JPG",
            "mask": "",
        },
        {
            "object": "candle",
            "split": "val",
            "label": "normal",
            "image": "candle/Data/Images/Normal/val.JPG",
            "mask": "",
        },
        {
            "object": "candle",
            "split": "test",
            "label": "normal",
            "image": "candle/Data/Images/Normal/test.JPG",
            "mask": "",
        },
        {
            "object": "candle",
            "split": "test",
            "label": "anomaly",
            "image": "candle/Data/Images/Anomaly/bad.JPG",
            "mask": "candle/Data/Masks/Anomaly/bad.png",
        },
    ]
    for row in rows:
        image_path = root / Path(row["image"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 12, 3), 127, dtype=np.uint8)).save(image_path)
        if row["mask"]:
            mask_path = root / Path(row["mask"])
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((8, 12), dtype=np.uint8)
            mask[2:5, 3:7] = 2
            Image.fromarray(mask).save(mask_path)
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    with split_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("object", "split", "label", "image", "mask")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_visa_preserves_official_splits_and_maps_masks(tmp_path):
    raw = tmp_path / "raw"
    split = tmp_path / "1cls.csv"
    prepared = tmp_path / "prepared" / "1cls"
    _make_raw_fixture(raw, split)

    manifest = prepare_dataset(raw, split, prepared, categories=["candle"])

    assert manifest["source_url"] == VISA_DATASET_URL
    assert manifest["rows_prepared"] == 4
    assert manifest["categories"] == [
        {
            "category": "candle",
            "train_normal": 1,
            "val_normal": 1,
            "val_anomaly": 0,
            "test_normal": 1,
            "test_anomaly": 1,
            "test_masks": 1,
        }
    ]
    assert len(load_train_images(prepared, "candle")) == 1
    assert validate_dataset(prepared, "candle") == ["candle"]
    items = load_test_items(prepared, "candle")
    assert len(items) == 2
    anomaly = next(item for item in items if item.label == "bad")
    assert anomaly.mask_path == (
        prepared / "candle" / "ground_truth" / "bad" / "bad_mask.png"
    )
    assert set(np.unique(anomaly.mask)) == {0, 255}
    assert (prepared / "candle" / "val" / "good" / "val.JPG").is_file()


@pytest.mark.parametrize(
    "mutation,expected_message",
    [
        ("missing-image", "Split CSV references missing file"),
        ("invalid-image", "Invalid image referenced"),
        ("missing-mask", "Split CSV references missing file"),
        ("invalid-label", "Unexpected label"),
    ],
)
def test_prepare_visa_rejects_corrupt_split_references(tmp_path, mutation, expected_message):
    raw = tmp_path / "raw"
    split = tmp_path / "1cls.csv"
    prepared = tmp_path / "prepared"
    _make_raw_fixture(raw, split)
    rows = list(csv.DictReader(split.open(encoding="utf-8")))
    if mutation == "missing-image":
        rows[0]["image"] = "candle/Data/Images/Normal/missing.JPG"
    elif mutation == "invalid-image":
        (raw / Path(rows[0]["image"])).write_bytes(b"invalid image")
    elif mutation == "missing-mask":
        rows[-1]["mask"] = "candle/Data/Masks/Anomaly/missing.png"
    else:
        rows[0]["label"] = "defective"
    with split.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("object", "split", "label", "image", "mask")
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises((FileNotFoundError, ValueError), match=expected_message):
        prepare_dataset(raw, split, prepared, categories=["candle"])


def test_prepare_visa_refuses_overwriting_existing_prepared_data(tmp_path):
    raw = tmp_path / "raw"
    split = tmp_path / "1cls.csv"
    prepared = tmp_path / "prepared"
    _make_raw_fixture(raw, split)
    prepared.mkdir()
    marker = prepared / "user-data.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_dataset(raw, split, prepared, categories=["candle"])
    assert marker.read_text(encoding="utf-8") == "keep"


def test_extract_visa_archive_verifies_existing_extraction_without_overwrite(tmp_path):
    archive = tmp_path / "visa.tar"
    with tarfile.open(archive, "w") as output:
        for category in EXPECTED_VISA_CATEGORIES:
            member = tarfile.TarInfo(f"{category}/")
            member.type = tarfile.DIRTYPE
            output.addfile(member)
        contents = b"verified"
        member = tarfile.TarInfo("candle/example.txt")
        member.size = len(contents)
        output.addfile(member, io.BytesIO(contents))

    extracted = tmp_path / "visa_raw"
    first = extract_archive(archive, extracted)
    assert first["archive_member_count"] == len(EXPECTED_VISA_CATEGORIES) + 1
    assert first["extracted_file_count"] == 1
    assert first["existing_extraction_verified"] is False
    existing_file = extracted / "candle" / "example.txt"
    existing_file.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="does not match the archive"):
        extract_archive(archive, extracted)
    assert existing_file.read_bytes() == b"preserve"


def test_extract_visa_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "malicious.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 1
        output.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(tarfile.ReadError, match="Unsafe archive path"):
        extract_archive(archive, tmp_path / "visa_raw")
