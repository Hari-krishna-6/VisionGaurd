from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def log_experiment(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Optional[float]],
    artifact_paths: tuple[str | Path, ...] = (),
    tracking_uri: Optional[str] = None,
    experiment_name: str = "visionguard",
) -> None:
    if not tracking_uri:
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({str(key): str(value) for key, value in params.items()})
        valid_metrics = {
            str(key): float(value)
            for key, value in metrics.items()
            if value is not None
        }
        if valid_metrics:
            mlflow.log_metrics(valid_metrics)
        for artifact_path in artifact_paths:
            path = Path(artifact_path)
            if path.is_file():
                mlflow.log_artifact(str(path))
            elif path.is_dir():
                mlflow.log_artifacts(str(path))
            else:
                raise FileNotFoundError(f"MLflow artifact does not exist: {path}")
