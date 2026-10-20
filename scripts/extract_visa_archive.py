from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from visionguard.data import EXPECTED_VISA_CATEGORIES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, output_root: Path) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(f"VisA archive does not exist: {archive}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = output_root.parent
    members_count = 0
    extracted_bytes = 0
    archive_size = archive.stat().st_size
    archive_digest = _sha256(archive)

    with tarfile.open(archive, mode="r:") as source:
        members = source.getmembers()
        if not members:
            raise tarfile.ReadError("Archive contains no members")
        total_bytes = sum(member.size for member in members if member.isfile())
        free_bytes = shutil.disk_usage(stage_parent).free
        if total_bytes > free_bytes:
            raise OSError(
                f"Insufficient free space: archive expands to {total_bytes} bytes, "
                f"available {free_bytes} bytes"
            )
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise tarfile.ReadError(f"Unsafe archive path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise tarfile.ReadError(
                    f"Unsupported special/link member in archive: {member.name}"
                )

        output_exists = output_root.exists() and any(output_root.iterdir())
        if output_exists:
            for member in members:
                if not member.isfile():
                    continue
                relative = PurePosixPath(member.name)
                destination = output_root.resolve().joinpath(*relative.parts)
                if not destination.is_file() or destination.stat().st_size != member.size:
                    raise FileExistsError(
                        "Existing extraction does not match the archive; refusing "
                        f"to overwrite it: {destination}"
                    )
                archive_stream = source.extractfile(member)
                if archive_stream is None:
                    raise tarfile.ReadError(
                        f"Could not read archive member: {member.name}"
                    )
                with archive_stream:
                    archive_file_hash = _sha256_stream(archive_stream)
                with destination.open("rb") as existing_stream:
                    existing_hash = _sha256_stream(existing_stream)
                if existing_hash != archive_file_hash:
                    raise FileExistsError(
                        "Existing extraction content does not match the archive; "
                        f"refusing to overwrite it: {destination}"
                    )
                members_count += 1
                extracted_bytes += member.size
        else:
            with tempfile.TemporaryDirectory(
                prefix=f".{output_root.name}-extract-", dir=stage_parent
            ) as temporary:
                stage = Path(temporary) / output_root.name
                stage.mkdir()
                for member in members:
                    relative = PurePosixPath(member.name)
                    destination = stage.joinpath(*relative.parts)
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    file_stream = source.extractfile(member)
                    if file_stream is None:
                        raise tarfile.ReadError(
                            f"Could not read archive member: {member.name}"
                        )
                    with file_stream, destination.open("xb") as output:
                        shutil.copyfileobj(file_stream, output, length=1024 * 1024)
                    extracted_bytes += member.size
                    members_count += 1
                    if members_count % 1000 == 0:
                        print(
                            f"Extracted {members_count} files "
                            f"({extracted_bytes / 1024**3:.2f} GiB)"
                        )
                resolved_output = output_root.resolve()
                if resolved_output.exists() and not any(resolved_output.iterdir()):
                    resolved_output.rmdir()
                shutil.move(str(stage), str(resolved_output))

    resolved_root = output_root.resolve()
    if (resolved_root / "VisA").is_dir():
        visa_root = resolved_root / "VisA"
    elif all((resolved_root / category).is_dir() for category in EXPECTED_VISA_CATEGORIES):
        visa_root = resolved_root
    else:
        raise tarfile.ReadError(
            f"Extracted archive lacks expected VisA categories: {resolved_root}"
        )
    return {
        "archive": str(archive.resolve()),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_digest,
        "archive_member_count": len(members),
        "extracted_file_count": members_count,
        "extracted_bytes": extracted_bytes,
        "extracted_root": str(output_root.resolve()),
        "raw_dataset_root": str(visa_root.resolve()),
        "existing_extraction_verified": output_exists,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely extract the official VisA tar archive."
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("data/downloads/VisA_20220922.tar")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/visa_raw"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("reports/visa_archive_integrity.json")
    )
    args = parser.parse_args()
    result = extract_archive(args.archive, args.output_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
