from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visionguard.config import load_config_from_env
from visionguard.data import load_test_items
from visionguard.features import preprocess_image, resize_anomaly_map
from visionguard.mlflow_utils import log_experiment
from visionguard.model import PatchCoreModel
from visionguard.visualization import generate_heatmap


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _process_memory_bytes() -> int | None:
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    try:
        import resource

        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak_rss if platform.system() == "Windows" else peak_rss * 1024
    except ImportError:
        return None


def _cpu_model() -> str:
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def run_benchmark(
    model_path: Path,
    image_paths: list[Path],
    output_path: Path,
    warmup: int = 20,
    samples: int = 100,
    run_name: str | None = None,
) -> dict:
    if not image_paths:
        raise ValueError("No benchmark images were found; provide a real dataset.")
    if warmup < 0 or samples < 1:
        raise ValueError("warmup must be non-negative and samples must be positive.")

    config = load_config_from_env()
    load_start = time.perf_counter()
    model = PatchCoreModel.load(model_path, device=config.device)
    model_load_seconds = time.perf_counter() - load_start
    device = model.feature_extractor.device
    selected_paths = [
        image_paths[index % len(image_paths)] for index in range(warmup + samples)
    ]
    phase_times: dict[str, list[float]] = {
        "preprocessing": [],
        "backbone_feature_extraction": [],
        "nearest_neighbor_search": [],
        "map_upsampling": [],
        "heatmap_rendering": [],
        "end_to_end_prediction": [],
    }
    cuda_peak_bytes = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def measure(path: Path, collect: bool) -> None:
        with Image.open(path) as source:
            image = source.convert("RGB")
        start = time.perf_counter()
        tensor = preprocess_image(image, model.input_size)
        if collect:
            phase_times["preprocessing"].append((time.perf_counter() - start) * 1000)

        _sync(device)
        start = time.perf_counter()
        embeddings, (height, width) = model.feature_extractor.extract(tensor)
        _sync(device)
        if collect:
            phase_times["backbone_feature_extraction"].append(
                (time.perf_counter() - start) * 1000
            )

        start = time.perf_counter()
        distances = model._nearest_patch_distances(embeddings)
        if collect:
            phase_times["nearest_neighbor_search"].append(
                (time.perf_counter() - start) * 1000
            )

        start = time.perf_counter()
        if distances.size != height * width:
            raise ValueError(
                f"Got {distances.size} scores for feature-map shape {(height, width)}"
            )
        patch_map = distances.reshape((height, width))
        raw_map = resize_anomaly_map(
            patch_map,
            (image.height, image.width),
            interpolation=model.config.anomaly_map_upsampling,
        )
        if collect:
            phase_times["map_upsampling"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        generate_heatmap(raw_map, (image.height, image.width))
        if collect:
            phase_times["heatmap_rendering"].append((time.perf_counter() - start) * 1000)

        _sync(device)
        start = time.perf_counter()
        model.predict_image(path)
        _sync(device)
        if collect:
            phase_times["end_to_end_prediction"].append(
                (time.perf_counter() - start) * 1000
            )

    for path in selected_paths[:warmup]:
        measure(path, collect=False)
    for path in selected_paths[warmup:]:
        measure(path, collect=True)
    if device.type == "cuda":
        cuda_peak_bytes = int(torch.cuda.max_memory_allocated(device))

    bank_size = int(model.memory_bank.nbytes)
    end_to_end = phase_times["end_to_end_prediction"]
    with Image.open(image_paths[0]) as source_image:
        source_dimensions = {
            "width": source_image.width,
            "height": source_image.height,
        }
    result = {
        "status": "measured",
        "dataset": str(config.dataset_root),
        "dataset_type": (
            "VisA"
            if "visa" in str(config.dataset_root).replace("\\", "/").lower()
            else "MVTec AD"
        ),
        "category": model.config.category,
        "image_count": samples,
        "warmup_count": warmup,
        "model_path": str(model_path),
        "device": str(device),
        "image_score_aggregation": model.config.image_score_aggregation,
        "image_score_top_k_fraction": model.config.image_score_top_k_fraction,
        "anomaly_map_upsampling": model.config.anomaly_map_upsampling,
        "model_load_seconds": model_load_seconds,
        "model_artifact_bytes": model_path.stat().st_size,
        "memory_bank_bytes": bank_size,
        "cpu_peak_working_set_bytes": _process_memory_bytes(),
        "cuda_peak_allocated_bytes": cuda_peak_bytes,
        "batch_size": 1,
        "input_dimensions": [model.input_size, model.input_size],
        "source_image_dimensions": source_dimensions,
        "cpu_model": _cpu_model(),
        "throughput_images_per_second": (
            samples / (sum(end_to_end) / 1000) if sum(end_to_end) else 0.0
        ),
        "timings": {name: _summary(values) for name, values in phase_times.items()},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    benchmark_metrics = {
        "model_load_seconds": model_load_seconds,
        "throughput_images_per_second": result["throughput_images_per_second"],
    }
    for phase, timing in result["timings"].items():
        benchmark_metrics[f"{phase}_mean_ms"] = timing["mean_ms"]
        benchmark_metrics[f"{phase}_median_ms"] = timing["median_ms"]
        benchmark_metrics[f"{phase}_p90_ms"] = timing["p90_ms"]
        benchmark_metrics[f"{phase}_p95_ms"] = timing["p95_ms"]
    if result["cpu_peak_working_set_bytes"] is not None:
        benchmark_metrics["cpu_peak_working_set_bytes"] = result[
            "cpu_peak_working_set_bytes"
        ]
    if cuda_peak_bytes is not None:
        benchmark_metrics["cuda_peak_allocated_bytes"] = cuda_peak_bytes
    log_experiment(
        run_name=run_name or (
            f"{model.config.category}-benchmark"
            if model.config.image_score_aggregation == "max"
            else (
                f"{model.config.category}-benchmark-"
                f"{model.config.image_score_aggregation}-"
                f"{round(model.config.image_score_top_k_fraction * 100)}pct"
            )
        ),
        params={
            "dataset_type": result["dataset_type"],
            "dataset_root": str(config.dataset_root),
            "dataset_identity": str(Path(config.dataset_root).resolve()),
            "category": model.config.category,
            "model_path": str(model_path),
            "device": str(device),
            "image_score_aggregation": model.config.image_score_aggregation,
            "image_score_top_k_fraction": (
                model.config.image_score_top_k_fraction
            ),
            "anomaly_map_upsampling": model.config.anomaly_map_upsampling,
            "cpu_model": result["cpu_model"],
            "warmup_count": warmup,
            "sample_count": samples,
            "batch_size": 1,
            "input_dimensions": f"{model.input_size}x{model.input_size}",
            "source_image_dimensions": (
                f"{source_dimensions['width']}x{source_dimensions['height']}"
            ),
            "model_artifact_bytes": result["model_artifact_bytes"],
            "memory_bank_bytes": bank_size,
            "sample_ratio": model.sample_ratio,
            "random_seed": model.random_seed,
            **result["software"],
        },
        metrics=benchmark_metrics,
        artifact_paths=(output_path,),
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.mlflow_experiment,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a trained model on real dataset test images."
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--category", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_results.json"))
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config = load_config_from_env()
    category = args.category or config.category
    dataset_root = args.dataset_root or Path(config.dataset_root)
    model_path = args.model or Path(config.model_path)
    paths = [
        item.image_path
        for item in load_test_items(dataset_root, category)
    ]
    result = run_benchmark(
        model_path,
        paths,
        args.output,
        args.warmup,
        args.samples,
        run_name=args.run_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
