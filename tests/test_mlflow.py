from __future__ import annotations

import mlflow

from visionguard.mlflow_utils import log_experiment


def test_mlflow_logger_records_parameters_metrics_and_artifacts(tmp_path):
    artifact = tmp_path / "model.txt"
    artifact.write_text("test artifact", encoding="utf-8")
    tracking_uri = (tmp_path / "runs").as_uri()

    log_experiment(
        run_name="test-run",
        params={"category": "tile", "sample_ratio": 0.01},
        metrics={"image_auroc": 0.75, "pixel_auroc": None},
        artifact_paths=(artifact,),
        tracking_uri=tracking_uri,
        experiment_name="visionguard-tests",
    )

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("visionguard-tests")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.run_name == "test-run"
    assert run.data.params["category"] == "tile"
    assert run.data.metrics["image_auroc"] == 0.75
    assert "pixel_auroc" not in run.data.metrics
    assert any(item.path == "model.txt" for item in client.list_artifacts(run.info.run_id))
