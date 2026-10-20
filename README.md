# VisionGuard MLOps

VisionGuard is an existing PatchCore-style anomaly-detection service for MVTec
AD and Amazon's VisA 1-class split, implemented with PyTorch, FastAPI, and
MLflow. Each model is trained for one product category using only normal
`train/good` images from the official training split. Synthetic data is
available only through explicit smoke-test commands; it is not real-dataset
evidence.

## Architecture

```text
normal train/good images
    -> ImageNet-preprocessed 224x224 tensors
    -> pretrained TorchVision Wide-ResNet50-2 layer2 and layer3 maps
    -> bilinear spatial alignment and 1536-channel concatenation
    -> deterministic greedy farthest-point coreset
    -> exact nearest-neighbour patch distances in bounded tiles
    -> image score + spatially reconstructed raw anomaly map
    -> calibrated image/pixel thresholds and test-split evaluation
    -> FastAPI prediction and uniquely named visual artifacts
```

Layer2 and layer3 activations are captured independently by direct module
execution, not stored by class name. At the default input size, the combined
feature grid is 28x28, with 1536 feature values per patch. The map's actual
height and width are used for reshape; square grids are not assumed.

The coreset greedily selects farthest points from a deterministic candidate
pool. The pool is capped at 20,000 candidates and projected to at most 128
dimensions when needed; returned vectors are the original feature vectors.
Selection is deterministic for a fixed seed. Exact nearest-neighbour distances
are computed in bounded query/bank tiles, not an all-pairs tensor.

## Requirements and setup

- Python 3.12; `requirements.txt` pins PyTorch 2.5.1 and TorchVision 0.20.1
  with the other runtime packages.
- CPU inference is the default. CUDA is used only when configured and supported
  by the installed PyTorch build and host.
- Initial training requires pretrained Wide-ResNet50-2 weights
  `Wide_ResNet50_2_Weights.IMAGENET1K_V2` from TorchVision. The weight file is
  about 263 MiB and may be downloaded by TorchVision during explicit training.
  TorchVision weight URL:
  [wide_resnet50_2-9ba9bcbe.pth](https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth).
  Training must not silently fall back to random weights.
- The saved NPZ model bundles the backbone state, memory bank, thresholds, and
  configuration. API startup does not need to download backbone weights.
- Full-category training stores extracted training features before coreset
  selection. RAM use depends on category size; check available resources before
  training all categories.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## VisA download and preparation

The VisA dataset and Amazon SPot-the-Difference repository are the primary
real-data pipeline in this run:

- Archive: [VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)
- Official repository: [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
  attribution is required and changes must be indicated.

Dataset attribution: Amazon Science, VisA dataset, distributed with the
[SPot-the-Difference repository](https://github.com/amazon-science/spot-diff).

PowerShell commands to resume/download, hash, safely extract, and prepare with
the official `split_csv/1cls.csv` protocol:

```powershell
New-Item -ItemType Directory -Force data\downloads | Out-Null
curl.exe --location --fail --retry 5 --retry-all-errors --retry-delay 5 `
  --continue-at - `
  https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar `
  --output data\downloads\VisA_20220922.tar
Get-FileHash data\downloads\VisA_20220922.tar -Algorithm SHA256
python -m scripts.extract_visa_archive
python -m scripts.prepare_visa
```

Raw source files remain in `data/visa_raw/`; the prepared 1-class dataset is
under `data/visa_prepared/1cls/<category>/`. The official 1-class CSV has
10,821 records: 8,659 train-normal, 962 test-normal, and 1,200 test-anomaly
images. It defines no validation split; preparation retains the official
train/test split and does not create one. Defect masks are binarized to 0/255
and matched by image stem. The manifest records archive hash, source,
repository revision, license, per-category split counts, and preparation.

The reproduction commands for candle (category first, then all categories)
are:

```powershell
$env:VISIONGUARD_DATASET_ROOT = "data\visa_prepared\1cls"
$env:VISIONGUARD_CATEGORY = "candle"
$env:VISIONGUARD_MODEL_PATH = "artifacts\model.npz"
$env:VISIONGUARD_SAMPLE_RATIO = "0.001"
$env:VISIONGUARD_RANDOM_SEED = "42"
$env:VISIONGUARD_DEVICE = "cpu"
$env:VISIONGUARD_MLFLOW_TRACKING_URI = "file:///G:/Projects/VisionGaurd/mlruns"
$env:VISIONGUARD_MLFLOW_EXPERIMENT = "visionguard-visa"
python -m scripts.train
python -m scripts.calibrate
python -m scripts.evaluate --category candle
python -m scripts.benchmark --category candle --warmup 20 --samples 100
```

For all categories, run training, calibration, and evaluation with
`--all-categories`. Evaluation supports `--resume`: it keeps compatible
measured category results in the report, records progress after each category,
and retries only missing or failed categories. For example, to safely resume
an interrupted all-category evaluation without retraining:

```powershell
python -m scripts.train --all-categories
python -m scripts.calibrate --all-categories
python -m scripts.evaluate --all-categories --resume
```

Training all categories may require substantial CPU time and RAM. Candle's
normal training split has 900
images. The sample ratio above (0.001) is explicitly lower than the general
default to keep CPU nearest-neighbour inference feasible; it is recorded in
model metadata and MLflow.

Phase 2 scoring experiments use the pre-existing fitted memory banks and
write separate candidate artifacts/reports; baseline artifacts and reports
remain unchanged. The initial candidate is the mean of the top 1% of patch
distances (eight of 784 patches for a 224x224 input). Its image threshold is
recalibrated with normal `train/good` images only at the baseline 99th
percentile. Pixel maps are unchanged, and the candidate reuses the saved
normal-only pixel threshold. The experiment reports test metrics only after
the candidate choice is fixed:

```powershell
python -m scripts.experiment_score_aggregation `
  --dataset-root data\visa_prepared\1cls `
  --categories candle macaroni1 macaroni2 `
  --output-dir reports\experiments\patchcore_topk_mean_1pct_v1
python -m scripts.compare_phase2
```

### Phase 3 localization experiment

For 224x224 model input, Wide-ResNet50-2 produces layer2 at 28x28 with 512
channels and layer3 at 14x14 with 1024 channels. Layer3 is bilinearly aligned
to 28x28, then concatenated with layer2 to create 784 patch vectors of 1536
dimensions. The raw nearest-neighbor distances form a 28x28 map, which is
resized to the source image dimensions. Preprocessing first stretches each
source image to a square and bilinearly resizes it to 224x224; normalization
uses ImageNet channel means and standard deviations. Thus map interpolation is
only one possible localization limitation: source resizing and the stride-8
feature grid also constrain spatial detail.

The Phase 3 candidate keeps preprocessing, backbone features, patch
distances, memory bank, image-level scoring, and official split unchanged. It
changes only final raw-map upsampling from bilinear to nearest-neighbor. Its
pixel threshold is recalibrated at the existing 99.5th percentile using only
normal `train/good` maps. The image threshold remains the saved baseline
normal-training threshold. This does not create a finer semantic feature
grid; it tests whether bilinear smoothing of the existing grid affects pixel
metrics.

```powershell
$env:VISIONGUARD_DATASET_ROOT = "data\visa_prepared\1cls"
$env:VISIONGUARD_DEVICE = "cpu"
$env:VISIONGUARD_MLFLOW_TRACKING_URI = "file:///G:/Projects/VisionGaurd/mlruns"
$env:VISIONGUARD_MLFLOW_EXPERIMENT = "visionguard-visa"
python -m scripts.experiment_localization `
  --dataset-root data\visa_prepared\1cls `
  --categories candle macaroni1 macaroni2 `
  --output-dir reports\experiments\phase3_localization_v1 `
  --warmup 20 --samples 100
```

The script preserves baseline artifacts, stores candidate models, test-split
diagnostics and comparison reports separately, and creates distinct MLflow
runs for category evaluation plus isolated baseline/candidate CPU benchmarks.
Do not use test labels or masks to alter the candidate or calibrate its
thresholds.

#### Phase 3 measured result

The nearest-upsample candidate did not improve localization and was not
promoted. Image scores and image thresholds are unchanged, so image metrics
match the baseline exactly. Pixel metrics declined across all three tested
categories; macro pixel AUROC/AP changed from **0.9631/0.09145** to
**0.9310/0.07648**, and pixel recall/F1 changed from **0.7033/0.07264** to
**0.6198/0.07119**. Candidate pixel thresholds were recalibrated using only
normal training maps with the same 99.5th percentile. They were not selected
using test labels or masks.

| Category | Pixel AUROC baseline → nearest | Pixel AP baseline → nearest | Pixel recall baseline → nearest | Pixel F1 baseline → nearest |
|---|---:|---:|---:|---:|
| candle | 0.9672 → 0.9480 | 0.1489 → 0.1253 | 0.7442 → 0.6836 | 0.1314 → 0.1303 |
| macaroni1 | 0.9729 → 0.9397 | 0.0828 → 0.0626 | 0.7877 → 0.6943 | 0.0458 → 0.0465 |
| macaroni2 | 0.9493 → 0.9053 | 0.0427 → 0.0416 | 0.5779 → 0.4816 | 0.0408 → 0.0368 |

Matched CPU benchmarks used 20 warmups and 100 test images per category.
Mean baseline versus candidate latency was candle **70.81 vs. 68.66 ms**,
macaroni1 **69.01 vs. 67.01 ms**, and macaroni2 **68.87 vs. 65.33 ms**. Peak
inference-benchmark working sets were approximately **0.82 GB** in each
configuration. The timing comparison has no confidence intervals and is not
evidence of a model-quality improvement. Full image/pixel metrics, diagnostic
examples, measured durations, and benchmark distributions are in
`reports/experiments/phase3_localization_v1/`.

### Phase 2 result (measured, exploratory)

The fixed top-1%-mean candidate was measured on the official test split for
candle, macaroni1, and macaroni2. Across those three categories, image AUROC
and average precision changed from 0.8690/0.8555 to 0.8771/0.8679, while
thresholded image recall/F1 changed from 0.2633/0.3756 to 0.2300/0.3597.
Macaroni2 regressed in AUROC, AP, recall, and F1. Pixel maps are identical by
design, so pixel metrics did not change. The candidate is slower in the matched
CPU benchmark below. It was not expanded to other categories or promoted; the
baseline remains the API model. These test-split measurements are exploratory
reporting, not permission to retune against test labels.

| Category | Image AUROC baseline → candidate | Image AP baseline → candidate | Image recall baseline → candidate | Image F1 baseline → candidate |
|---|---:|---:|---:|---:|
| candle | 0.8719 → 0.9314 | 0.8171 → 0.9311 | 0.0300 → 0.1700 | 0.0577 → 0.2906 |
| macaroni1 | 0.9184 → 0.9335 | 0.9153 → 0.9267 | 0.4100 → 0.3600 | 0.5655 → 0.5217 |
| macaroni2 | 0.8168 → 0.7664 | 0.8342 → 0.7458 | 0.3500 → 0.1600 | 0.5036 → 0.2667 |

Matched CPU benchmark commands (20 warmups and 100 measured candle test
images in each run):

```powershell
python -m scripts.benchmark --model artifacts\model.npz --dataset-root data\visa_prepared\1cls --category candle --warmup 20 --samples 100 --output reports\experiments\phase2_baseline_benchmark.json --run-name phase2-baseline-max-candle
python -m scripts.benchmark --model artifacts\experiments\patchcore_topk_mean_1pct_v1\candle\model.npz --dataset-root data\visa_prepared\1cls --category candle --warmup 20 --samples 100 --output reports\experiments\patchcore_topk_mean_1pct_v1\benchmark_results.json
```

Both runs used the Intel Core Ultra 7 270K Plus CPU, Python 3.12.13,
PyTorch 2.5.1+cpu, and NumPy 1.26.4. Baseline mean/median/p95 latency was
62.15/62.16/66.34 ms (16.09 images/s); candidate latency was
63.92/64.22/70.94 ms (15.64 images/s). Peak process working sets were
820,477,952 and 820,277,248 bytes, respectively. These are single-process
measurements without confidence intervals; CUDA was unavailable.

## Official MVTec AD acquisition and layout

Use the official source:
[MVTec AD download page](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Read the current download instructions and terms there before acquiring data.
This workspace could not reach that page during the current validation, so the
current terms and download procedure remain unverified. No download or mirror
was attempted. Dataset data must not be committed.

After acquiring and extracting the dataset, set `VISIONGUARD_DATASET_ROOT` to
the extracted `mvtec_ad` directory (default `data/mvtec_ad`). Preserve the
original split and names:

```text
data/mvtec_ad/
  bottle/
    train/good/*.png
    test/
      good/*.png
      broken_large/*.png
      ...
    ground_truth/
      broken_large/<image-stem>_mask.png
      ...
  cable/
    ...
```

The standard MVTec categories are `bottle`, `cable`, `capsule`, `carpet`,
`grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`,
`toothbrush`, `transistor`, `wood`, and `zipper`. Every test defect directory
other than `good` is anomalous. Masks are matched under
`ground_truth/<defect-directory>/<image-stem>_mask.png`.

Validate one category or the complete dataset and write a JSON manifest:

```powershell
python -m scripts.validate_dataset --dataset-root data/mvtec_ad --category bottle
python -m scripts.validate_dataset --dataset-root data/mvtec_ad
```

The manifest includes discovered/missing categories, split counts, malformed
images, and missing masks. A missing anomaly mask does not fabricate pixel
metrics; pixel metrics are withheld and the missing count is reported.

## Training, calibration, and evaluation

Configure `VISIONGUARD_DATASET_ROOT`, `VISIONGUARD_CATEGORY` (default
`bottle`), and optional `VISIONGUARD_MODEL_PATH` before running:

```powershell
$env:VISIONGUARD_DATASET_ROOT = "data/mvtec_ad"
$env:VISIONGUARD_CATEGORY = "bottle"
python -m scripts.train
python -m scripts.calibrate
python -m scripts.evaluate --category bottle
```

Training fits the memory bank from `train/good` only and writes
`artifacts/model.npz` plus `artifacts/model.metadata.json`. Calibration uses
normal `train/good` scores: the 99th percentile for image scores and the 99.5th
percentile for pixel scores. Test labels are never used for threshold
selection. The thresholds are saved in the model artifact. These are
training-set calibration thresholds, not thresholds selected from a separate
validation split; that limitation should be considered when interpreting
metrics.

To fit/evaluate all discovered categories, one model per category:

```powershell
python -m scripts.train --all-categories
python -m scripts.calibrate --all-categories
python -m scripts.evaluate --all-categories
```

Training writes each model beneath `artifacts/<category>/` in all-category
mode. The all-category calibration command uses those category-specific model
paths. Evaluation reports failures instead of silently omitting categories.

The evaluator reports image-level AUROC, average precision, precision, recall,
F1, confusion matrix, FPR/FNR; pixel-level counterparts when masks are
complete; missing-mask state; per-category results; and an equal-weight
macro-average. Undefined metrics are null. Reports include test-image
predictions, representative original/mask/heatmap/overlay images, and raw
anomaly score arrays:

- `reports/evaluation_results.json`
- `reports/per_category_metrics.csv`
- `reports/image_predictions.csv`
- `reports/evaluation_report.md`
- `reports/examples/<category>/`

Verified VisA run: all 12 categories completed on the official test split
(2,162 test images total); all 1,200 anomalous images had matched,
dimension-aligned masks. Calibration used normal `train/good` images only.
The macro-average image AUROC/AP was 0.8890/0.8995, and pixel AUROC/AP was
0.9563/0.3186. Threshold-based macro image recall was 0.4208; this is a
training-normal 99th-percentile threshold, not a validation-optimized
operating point. Per-category AUROC/AP:

| Category | Image AUROC | Image AP | Pixel AUROC | Pixel AP |
|---|---:|---:|---:|---:|
| candle | 0.8719 | 0.8171 | 0.9672 | 0.1489 |
| capsules | 0.7578 | 0.8645 | 0.9501 | 0.4399 |
| cashew | 0.9126 | 0.9568 | 0.9361 | 0.4213 |
| chewinggum | 0.9880 | 0.9944 | 0.9811 | 0.5042 |
| fryum | 0.9354 | 0.9708 | 0.9015 | 0.2881 |
| macaroni1 | 0.9184 | 0.9153 | 0.9729 | 0.0828 |
| macaroni2 | 0.8168 | 0.8342 | 0.9493 | 0.0427 |
| pcb1 | 0.8586 | 0.8138 | 0.9818 | 0.5931 |
| pcb2 | 0.8180 | 0.8190 | 0.9421 | 0.1676 |
| pcb3 | 0.8574 | 0.8694 | 0.9730 | 0.3273 |
| pcb4 | 0.9442 | 0.9448 | 0.9491 | 0.3482 |
| pipe_fryum | 0.9884 | 0.9942 | 0.9715 | 0.4596 |

Run the measured CPU/GPU benchmark on actual category test images:

```powershell
python -m scripts.benchmark --category candle --warmup 20 --samples 100
```

The benchmark records cold model-artifact load time, warmup and sample counts,
mean/median/p90/p95 single-image latency, phase times, throughput, peak process
working set, GPU memory when applicable, model/memory-bank size, image
dimensions, and software/device details. It requires a real dataset and a
trained model. It does not benchmark ONNX Runtime; ONNX support is not
implemented. The measured candle CPU run used 20 warmups and 100 samples:
mean/median/p95 end-to-end latency 68.19/67.58/73.55 ms, p90 71.72 ms,
throughput 14.66 images/s, peak process working set 821,108,736 bytes, and
model loading 1.42 s. Hardware: Intel Core Ultra 7 270K Plus; CUDA was
unavailable.

## Synthetic smoke tests (not benchmark results)

Synthetic data is temporary and explicitly opt-in:

```powershell
python -m scripts.train --smoke-test
python -m scripts.calibrate --smoke-test
python -m scripts.evaluate --smoke-test
```

These only validate code paths. They do not represent VisA or MVTec AD, model
quality, real-data evaluation, or performance.

## API and client

The API loads the configured, trained artifact at startup. If no artifact,
fitted memory bank, or calibrated image threshold is available, `/predict`
returns an actionable HTTP 503 response. The API does not train or generate a
fake model.

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
python -m scripts.predict "path\to\existing-image.png"
```

`GET /health` reports whether an artifact loaded. `POST /predict` accepts PNG
and JPEG files; uploads are capped at 10 MiB by default, with an additional
40-megapixel limit. Set `VISIONGUARD_MAX_UPLOAD_BYTES` to change the byte
limit. Responses include image score, calibrated status, threshold, dimensions,
and URLs for original/heatmap/overlay/legend artifacts. Request-specific UUID
filenames prevent concurrent predictions from overwriting one another.
`scripts.predict` takes an existing image and can download the returned
artifacts; it never generates a substitute input image.

## MLflow

Training, calibration, and evaluation call the MLflow logger when
`VISIONGUARD_MLFLOW_TRACKING_URI` is set. Runs log measured durations/metrics,
category, dataset identity/type, split, model configuration, software
versions, and model/report artifacts. Synthetic runs are explicitly tagged;
they are not real-data runs.

```powershell
$env:VISIONGUARD_MLFLOW_TRACKING_URI = "file:///C:/path/to/mlruns"
$env:VISIONGUARD_MLFLOW_EXPERIMENT = "visionguard"
python -m scripts.train
python -m scripts.calibrate
python -m scripts.evaluate
mlflow ui --backend-store-uri "file:///C:/path/to/mlruns"
```

If the tracking URI is unset, MLflow tracking is disabled. The verified VisA
experiment is named `visionguard-visa`; its training, calibration, evaluation,
and benchmark runs log dataset identity, configuration, measured metrics, and
artifacts. Test-suite MLflow checks use a separate isolated temporary store.

## Docker

Train and calibrate on the host first. The Docker image uses a pinned Python
3.12 base image, installs the pinned CPU PyTorch/TorchVision pair, runs as a
non-root user, and includes a health check. The pretrained backbone state is
in the saved model artifact, so startup does not fetch weights.

```powershell
docker compose build api
docker compose up -d api
Invoke-RestMethod http://127.0.0.1:8000/health
# Upload a real VisA image and retrieve the returned artifact URLs:
curl.exe -F "file=@data\visa_prepared\1cls\candle\test\bad\014.JPG;type=image/jpeg" http://127.0.0.1:8000/predict
docker compose down
```

Compose maps `data/visa_prepared/1cls` read-only to `/data/visa_prepared/1cls`,
the candle model read-only to `/app/artifacts/model.npz`, and heatmap output
to `/app/artifacts/heatmaps`. It uses the candle category and CPU device.
Set `VISIONGUARD_MODEL_PATH`, `VISIONGUARD_DATASET_ROOT`,
`VISIONGUARD_CATEGORY`, and `VISIONGUARD_HEATMAP_DIR` to override paths.

## Tests, generated status reports, and limitations

```powershell
.\.venv\Scripts\python.exe -m pytest -q
python -m scripts.generate_status_reports
```

`generate_status_reports` updates the environment snapshot and validation log.
It deliberately preserves actual evaluation, prediction, dataset manifest,
and benchmark output files. The pipeline commands produce:

- `reports/dataset_manifest.json`
- `reports/evaluation_results.json`
- `reports/per_category_metrics.csv`
- `reports/benchmark_results.json`
- `reports/evaluation_report.md`
- `reports/environment.json`
- `reports/validation_log.md`

Known limitations: MVTec AD has not been downloaded or evaluated. The official
VisA 1-class split has no validation set, so thresholds are calibrated from
normal training images. The host is CPU-only; no CUDA result or confidence
intervals are reported. There is no API authentication, rate limiting,
production monitoring, or ONNX export/inference. This project makes no
production-readiness claim. See the generated evaluation, validation, and
benchmark reports for category-level completion and exact measurements.
