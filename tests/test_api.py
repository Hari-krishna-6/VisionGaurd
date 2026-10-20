from __future__ import annotations

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from visionguard.config import ModelConfig
from visionguard.model import PatchCoreModel

client = TestClient(app)


class LightweightFeatureExtractor:
    def extract(self, image_tensor):
        assert image_tensor.ndim == 4
        return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32), (2, 2)


def _png_bytes(width: int = 24, height: int = 16) -> bytes:
    image = Image.new("RGB", (width, height), color=(120, 140, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fitted_lightweight_model() -> PatchCoreModel:
    model = PatchCoreModel.__new__(PatchCoreModel)
    model.config = ModelConfig(image_size=32)
    model.input_size = 32
    model.feature_extractor = LightweightFeatureExtractor()
    model.memory_bank = np.zeros((2, 2), dtype=np.float32)
    model.threshold = 0.5
    return model


def test_api_health(monkeypatch):
    monkeypatch.setattr(app.state, "model", None)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is False


def test_api_rejects_unsupported_file_type():
    response = client.post("/predict", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_api_rejects_malformed_image():
    response = client.post(
        "/predict",
        files={"file": ("bad.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400
    assert "Malformed image" in response.json()["detail"]


def test_api_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(app.state.config, "max_upload_bytes", 8)
    response = client.post(
        "/predict",
        files={"file": ("large.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 413


def test_api_reports_missing_model_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(app.state, "model", None)
    monkeypatch.setattr(app.state, "model_load_error", None)
    monkeypatch.setattr(app.state.config, "model_path", str(tmp_path / "missing.npz"))
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert "python -m scripts.train" in response.json()["detail"]


def test_api_rejects_unfitted_model(monkeypatch):
    unfitted = PatchCoreModel.__new__(PatchCoreModel)
    unfitted.input_size = 32
    unfitted.feature_extractor = LightweightFeatureExtractor()
    unfitted.memory_bank = None
    unfitted.threshold = None
    monkeypatch.setattr(app.state, "model", unfitted)

    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert "no fitted memory bank" in response.json()["detail"]


def test_api_rejects_uncalibrated_fitted_model(monkeypatch):
    model = _fitted_lightweight_model()
    model.threshold = None
    monkeypatch.setattr(app.state, "model", model)
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert "no calibrated anomaly threshold" in response.json()["detail"]


def test_api_prediction_returns_accessible_unique_artifacts(monkeypatch):
    monkeypatch.setattr(app.state, "model", _fitted_lightweight_model())
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["image_dimensions"] == {"width": 24, "height": 16}
    assert payload["status"] == "anomalous"
    assert {"original", "heatmap", "overlay", "legend"} == set(payload["heatmap_artifacts"])
    overlay_url = payload["heatmap_artifacts"]["overlay"]
    assert client.get(overlay_url).status_code == 200

    second = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert second.status_code == 200
    assert (
        Path(payload["heatmap_artifacts"]["overlay"]).name
        != Path(second.json()["heatmap_artifacts"]["overlay"]).name
    )


def test_api_concurrent_requests_write_unique_artifact_names(monkeypatch, tmp_path):
    monkeypatch.setattr(app.state, "model", _fitted_lightweight_model())
    monkeypatch.setattr(app.state.config, "heatmap_dir", str(tmp_path))

    def predict_once():
        with TestClient(app) as local_client:
            return local_client.post(
                "/predict",
                files={"file": ("sample.png", _png_bytes(), "image/png")},
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda _index: predict_once(), range(4)))
    assert all(response.status_code == 200 for response in responses)
    artifact_names = {
        Path(response.json()["heatmap_artifacts"]["original"]).name
        for response in responses
    }
    assert len(artifact_names) == len(responses)


def test_api_openapi_documents_predict_route():
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/predict" in schema.json()["paths"]


def test_api_loads_saved_fitted_model_artifact(monkeypatch, tmp_path):
    config = ModelConfig(model_name="untrained-test-backbone")
    model = PatchCoreModel(config=config, input_size=32, sample_ratio=1.0)
    model.memory_bank = np.zeros((1, 512 + 1024), dtype=np.float32)
    model.threshold = 1.0
    artifact = tmp_path / "fitted-model.npz"
    model.save(artifact)

    monkeypatch.setattr(app.state, "model", None)
    monkeypatch.setattr(app.state, "model_load_error", None)
    monkeypatch.setattr(app.state.config, "model_path", str(artifact))
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["threshold"] == 1.0
