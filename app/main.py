from __future__ import annotations

import io
import logging
import threading
import urllib.error
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from visionguard.config import load_config_from_env
from visionguard.model import PatchCoreModel
from visionguard.visualization import save_heatmap_artifacts

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    if application.state.model is None and application.state.model_load_error is None:
        _load_configured_model()
    yield


app = FastAPI(title="VisionGuard MLOps API", lifespan=lifespan)
app.state.config = load_config_from_env()
app.state.model = None
app.state.model_load_error = None
app.state.model_lock = threading.Lock()
app.mount(
    "/artifacts",
    StaticFiles(directory=app.state.config.heatmap_dir, check_dir=False),
    name="artifacts",
)


def _load_configured_model() -> None:
    model_path = Path(app.state.config.model_path)
    if not model_path.is_file():
        app.state.model_load_error = (
            f"No trained model artifact is available at '{model_path}'. "
            "Run 'python -m scripts.train' with the target dataset first, or set VISIONGUARD_MODEL_PATH."
        )
        return
    try:
        app.state.model = PatchCoreModel.load(model_path, device=app.state.config.device)
        app.state.model_load_error = None
    except (
        OSError,
        ValueError,
        KeyError,
        EOFError,
        RuntimeError,
        urllib.error.URLError,
        zipfile.BadZipFile,
    ) as exc:
        app.state.model_load_error = f"Could not load model artifact '{model_path}': {exc}"
        logger.exception(app.state.model_load_error)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "VisionGuard PatchCore",
        "device": app.state.config.device,
        "model_loaded": app.state.model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file is required")
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only PNG/JPEG images are supported")

    data = await file.read(app.state.config.max_upload_bytes + 1)
    if len(data) > app.state.config.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {app.state.config.max_upload_bytes}-byte limit",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            width, height = probe.size
            probe.verify()
        if image_format not in {"PNG", "JPEG"}:
            raise HTTPException(status_code=400, detail="Uploaded content is not PNG or JPEG")
        if width < 1 or height < 1:
            raise HTTPException(status_code=400, detail="Uploaded image has invalid dimensions")
        if width * height > 40_000_000:
            raise HTTPException(status_code=413, detail="Uploaded image exceeds the 40-megapixel limit")
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f"Malformed image upload: {exc}") from exc

    if app.state.model is None:
        with app.state.model_lock:
            if app.state.model is None and app.state.model_load_error is None:
                _load_configured_model()
    if app.state.model is None:
        detail = app.state.model_load_error or (
            f"No trained model artifact is available at '{app.state.config.model_path}'. "
            "Run 'python -m scripts.train' with the target dataset first, or set VISIONGUARD_MODEL_PATH."
        )
        raise HTTPException(status_code=503, detail=detail)
    if app.state.model.memory_bank is None:
        raise HTTPException(
            status_code=503,
            detail="The loaded model has no fitted memory bank; train and save a model before prediction.",
        )
    if app.state.model.threshold is None:
        raise HTTPException(
            status_code=503,
            detail="The loaded model has no calibrated anomaly threshold; retrain it or configure VISIONGUARD_THRESHOLD.",
        )

    try:
        score, anomaly_map = app.state.model.predict_image(image)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Model is not ready for prediction: {exc}") from exc

    artifact_dir = Path(app.state.config.heatmap_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    artifacts = save_heatmap_artifacts(image, anomaly_map, artifact_dir, stem=request_id)
    artifact_urls = {
        name: f"/artifacts/{Path(path).name}"
        for name, path in artifacts.items()
    }
    threshold = app.state.model.threshold
    return {
        "image_score": score,
        "status": "anomalous" if threshold is not None and score >= threshold else "normal",
        "threshold": threshold,
        "image_dimensions": {"width": width, "height": height},
        "heatmap_artifacts": artifact_urls,
        "model": {"name": app.state.config.model_name, "device": app.state.config.device},
        "config": app.state.config.to_dict(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
