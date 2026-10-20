from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from urllib.parse import urljoin

import requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send an existing PNG/JPEG image to VisionGuard and download artifacts."
    )
    parser.add_argument("image", type=Path, help="Path to an existing input image.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/predictions"))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input image does not exist: {args.image}")
    content_type = mimetypes.guess_type(args.image.name)[0]
    if content_type not in {"image/png", "image/jpeg"}:
        parser.error("Input image must have a PNG or JPEG extension.")

    endpoint = args.api_url.rstrip("/") + "/predict"
    with args.image.open("rb") as handle:
        response = requests.post(
            endpoint,
            files={"file": (args.image.name, handle, content_type)},
            timeout=60,
        )
    response.raise_for_status()
    payload = response.json()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for url in payload.get("heatmap_artifacts", {}).values():
        artifact_response = requests.get(urljoin(args.api_url.rstrip("/") + "/", url), timeout=30)
        artifact_response.raise_for_status()
        filename = Path(url).name
        (args.output_dir / filename).write_bytes(artifact_response.content)
    print(json.dumps(payload, indent=2))
    print(f"Downloaded prediction artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
