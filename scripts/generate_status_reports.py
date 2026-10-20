from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASET = ROOT / "data" / "visa_prepared" / "1cls"
MODEL = ROOT / "artifacts" / "model.npz"


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _memory_bytes() -> dict[str, int | None]:
    if platform.system() != "Windows":
        return {"total": None, "available": None}

    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"total": None, "available": None}
    return {
        "total": int(status.ullTotalPhys),
        "available": int(status.ullAvailPhys),
    }


def _docker_status() -> dict:
    docker = shutil.which("docker")
    if docker is None:
        return {"client_found": False, "engine_available": False}
    client_result = subprocess.run(
        [docker, "--version"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "client_found": True,
            "client_version": client_result.stdout.strip(),
            "engine_available": False,
            "error": "docker info timed out",
        }
    return {
        "client_found": True,
        "client_version": client_result.stdout.strip(),
        "engine_available": result.returncode == 0,
        "server_version": result.stdout.strip() if result.returncode == 0 else None,
        "error": result.stderr.strip() if result.returncode else None,
    }


def _docker_runtime_status() -> tuple[dict | None, dict | None]:
    docker = shutil.which("docker")
    if docker is None:
        return None, None
    image_result = subprocess.run(
        [
            docker,
            "image",
            "inspect",
            "visiongaurd-api:latest",
            "--format",
            "{{.Id}}|{{.Size}}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    image = None
    if image_result.returncode == 0:
        image_id, image_size = image_result.stdout.strip().split("|", maxsplit=1)
        image = {"id": image_id, "size_bytes": int(image_size)}
    container_result = subprocess.run(
        [
            docker,
            "inspect",
            "--format",
            "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            "visiongaurd-api-1",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    container = None
    if container_result.returncode == 0:
        state, health = container_result.stdout.strip().split("|", maxsplit=1)
        container = {"state": state, "health": health}
    return image, container


def _mlflow_runs(tracking_uri: str, experiment_name: str) -> list[dict]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_ids = [
        experiment.experiment_id
        for experiment in client.search_experiments()
        if experiment.name == experiment_name
    ]
    runs = client.search_runs(
        experiment_ids,
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    ) if experiment_ids else []
    return [
        {
            "run_id": run.info.run_id,
            "name": run.data.tags.get("mlflow.runName"),
            "status": run.info.status,
            "category": run.data.params.get("category"),
            "dataset_type": run.data.params.get("dataset_type"),
            "sample_ratio": run.data.params.get("sample_ratio"),
            "metric_keys": sorted(run.data.metrics),
            "artifact_paths": [
                artifact.path for artifact in client.list_artifacts(run.info.run_id)
            ],
        }
        for run in runs
    ]


def _cpu_info() -> dict[str, str | int | None]:
    if platform.system() == "Windows":
        command = (
            "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1; "
            "[PSCustomObject]@{Name=$cpu.Name; Cores=$cpu.NumberOfCores; "
            "LogicalProcessors=$cpu.NumberOfLogicalProcessors} | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    "model": data.get("Name") or platform.processor() or platform.machine(),
                    "cores": data.get("Cores"),
                    "logical_processors": data.get("LogicalProcessors"),
                }
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
    return {
        "model": platform.processor() or platform.machine(),
        "cores": None,
        "logical_processors": None,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _existing_verification_record(path: Path) -> str | None:
    if not path.is_file():
        return None
    existing = path.read_text(encoding="utf-8")
    marker = "## Verification record"
    if marker not in existing:
        return None
    record = existing.split(marker, maxsplit=1)[1].strip()
    if not record or "Run and record the exact commands/results" in record:
        return None
    return f"{marker}\n\n{record}"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    manifest_path = REPORTS / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    disk = shutil.disk_usage(ROOT)
    cpu = _cpu_info()
    docker_image, docker_container = _docker_runtime_status()
    mlflow_uri = (ROOT / "mlruns").as_uri()
    mlflow_runs = _mlflow_runs(mlflow_uri, "visionguard-visa")
    environment = {
        "collected_at_utc": now,
        "workspace": str(ROOT),
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "memory_bytes": _memory_bytes(),
        "workspace_drive_bytes": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
        },
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dependencies": {
            name: _package_version(name)
            for name in (
                "numpy",
                "Pillow",
                "opencv-python",
                "scikit-learn",
                "fastapi",
                "uvicorn",
                "mlflow",
                "pytest",
                "requests",
            )
        },
        "docker": _docker_status(),
        "docker_image": docker_image,
        "docker_container": docker_container,
        "mlflow_tracking_uri": mlflow_uri,
        "mlflow_experiment": "visionguard-visa",
        "mlflow_runs": mlflow_runs,
        "dataset_root": str(DATASET),
        "dataset_present": DATASET.is_dir(),
        "dataset_identity": manifest.get("repository_url"),
        "archive_sha256": manifest.get("archive_integrity", {}).get("archive_sha256"),
        "trained_model_artifact": str(MODEL),
        "trained_model_artifact_present": MODEL.is_file(),
    }
    cached_weights = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / "wide_resnet50_2-9ba9bcbe.pth"
    )
    environment["cached_torchvision_weights"] = str(cached_weights)
    environment["cached_torchvision_weights_present"] = cached_weights.is_file()
    environment["cached_torchvision_weights_bytes"] = (
        cached_weights.stat().st_size if cached_weights.is_file() else None
    )
    _write_json(REPORTS / "environment.json", environment)

    evaluation_path = REPORTS / "evaluation_results.json"
    benchmark_path = REPORTS / "benchmark_results.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.is_file()
        else {"status": "not_run"}
    )
    benchmark = (
        json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark_path.is_file()
        else {"status": "not_run"}
    )
    category_counts = manifest.get("categories", [])
    validation_log_path = REPORTS / "validation_log.md"
    preserved_verification_record = _existing_verification_record(
        validation_log_path
    )
    verification_record = (
        preserved_verification_record
        or """## Verification record

Run and record the exact commands/results for tests, Python compilation, Docker
image build, and external API checks for this execution. No status is inferred
or fabricated by this report generator."""
    )
    validation_log = f"""# Validation log

Generated: {now}

## Dataset

- Source: {manifest.get('source_url')}
- Dataset identity: {manifest.get('repository_url')} at revision {manifest.get('repository_revision')}.
- License: {manifest.get('license', {}).get('name')} ({manifest.get('license', {}).get('url')}).
- Archive SHA-256: `{environment['archive_sha256']}`.
- Archive and preparation status: `{manifest.get('archive_integrity', {}).get('existing_extraction_verified')}`; {manifest.get('rows_prepared')} official 1-class rows prepared across {len(category_counts)} categories.
- Dataset root present: `{DATASET.is_dir()}`. Model artifact present: `{MODEL.is_file()}`.
- Official splits are preserved; calibration uses train normal images and no test labels.

## Evaluation and benchmark

- Real-data evaluation: `{evaluation.get('status')}` ({evaluation.get('dataset_type') or 'dataset type not recorded'}).
- Real-data benchmark: `{benchmark.get('status')}` ({benchmark.get('dataset_type') or 'dataset type not recorded'}).
- Reports are read as produced by their respective commands; this generator does not overwrite evaluation metrics, predictions, or benchmark measurements.

## Environment and Docker

- OS: {environment['operating_system']}; Python {environment['python']}; PyTorch {environment['torch']}; TorchVision {environment['torchvision']}.
- CPU: {cpu}; CUDA available: `{environment['cuda_available']}`.
- Docker engine: `{json.dumps(environment['docker'])}`.
- Docker image: `{json.dumps(docker_image)}`; container: `{json.dumps(docker_container)}`.
- MLflow experiment `visionguard-visa`: {len(mlflow_runs)} logged runs; statuses: `{', '.join(sorted({run['status'] for run in mlflow_runs})) or 'none'}`.

{verification_record}
"""
    validation_log_path.write_text(validation_log, encoding="utf-8")
    print(json.dumps({
        "reports_directory": str(REPORTS),
        "dataset_present": DATASET.is_dir(),
        "model_artifact_present": MODEL.is_file(),
        "evaluation_status": evaluation.get("status"),
        "benchmark_status": benchmark.get("status"),
        "docker_engine_available": environment["docker"]["engine_available"],
    }, indent=2))


if __name__ == "__main__":
    main()
