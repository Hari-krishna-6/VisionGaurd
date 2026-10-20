# Validation log

Generated: 2026-10-20T11:49:54.572780+00:00

## Dataset

- Source: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
- Dataset identity: https://github.com/amazon-science/spot-diff at revision 2a692ab575001cbde74d402d897a7286086c6199.
- License: Creative Commons Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/).
- Archive SHA-256: `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`.
- Archive and preparation status: `True`; 10821 official 1-class rows prepared across 12 categories.
- Dataset root present: `True`. Model artifact present: `True`.
- Official splits are preserved; calibration uses train normal images and no test labels.

## Evaluation and benchmark

- Real-data evaluation: `measured` (VisA).
- Real-data benchmark: `measured` (VisA).
- Reports are read as produced by their respective commands; this generator does not overwrite evaluation metrics, predictions, or benchmark measurements.

## Environment and Docker

- OS: Windows-11-10.0.26200-SP0; Python 3.12.13; PyTorch 2.5.1+cpu; TorchVision 0.20.1+cpu.
- CPU: {'model': 'Intel(R) Core(TM) Ultra 7 270K Plus', 'cores': 24, 'logical_processors': 24}; CUDA available: `False`.
- Docker engine: `{"client_found": true, "client_version": "Docker version 29.8.1, build 4a63305", "engine_available": true, "server_version": "29.8.1", "error": null}`.
- Docker image: `{"id": "sha256:243310d44bc649ebb753a8c63a20ab1c3618e3ae7998652418be0d15d9763439", "size_bytes": 3028461877}`; container: `{"state": "running", "health": "healthy"}`.
- MLflow experiment `visionguard-visa`: 55 logged runs; statuses: `FINISHED`.

## Verification record

### Dataset and evaluation

- `reports/evaluation_results.json`: `measured`; 12/12 categories completed, zero failures, zero pending categories, 2,162 official test images, and zero missing anomalous masks.
- The official test images and anomaly masks were matched by filename and checked for equal dimensions: 1,200 anomaly/mask pairs checked, zero alignment errors.
- Thresholds were loaded from each category artifact calibrated on normal `train/good` images only. No test labels were used for calibration.
- Macro image AUROC/AP: `0.88895979` / `0.89952682`; macro pixel AUROC/AP: `0.95630217` / `0.31864184`.
- The report preserves per-image predictions, category metrics, confusion matrices, and original/mask/heatmap/overlay/raw-score examples.

### Benchmark

- Command: `python -m scripts.benchmark --category candle --warmup 20 --samples 100`
- Device: CPU; CUDA unavailable. CPU: Intel(R) Core(TM) Ultra 7 270K Plus.
- 100 measured samples after 20 warmups; model load `1.4150 s`; end-to-end mean/median/p90/p95 `68.1916/67.5828/71.7155/73.5452 ms`; throughput `14.6646 images/s`.
- Peak process working set: `821108736` bytes; model artifact `258677895` bytes; memory bank `4337664` bytes.

### MLflow

- Experiment: `visionguard-visa`; 41 runs, all `FINISHED`.
- Logged coverage: 13 training, 13 calibration, 13 evaluation, and 2 benchmark runs. The extra training/calibration/evaluation runs are prior candle runs; all 12 categories have completed runs for each stage. Both benchmark runs are candle benchmarks.
- Every run group has measured metrics and artifacts. Per-category evaluation artifacts include `evaluation.json` and the corresponding model artifact.
- Added the missing absolute `dataset_identity` parameter to both existing benchmark runs in place; no benchmark run was duplicated. Future benchmark runs log this parameter directly.

### Tests and compilation

- Command: `.\.venv\Scripts\python.exe -m pytest -q`
- Result: `40 passed, 0 failed` in `29.31 s`; 18 dependency deprecation warnings (Starlette/AnyIO, MLflow `pkg_resources`, and Pydantic v1-style validators).
- Command: `.\.venv\Scripts\python.exe -m compileall -q app visionguard scripts tests`
- Result: `PASS`.

### Docker and external API

- Command: `docker compose config -q` — `PASS`.
- The verified image was reused; no rebuild was needed for evaluation/report/test changes. Image ID: `sha256:243310d44bc649ebb753a8c63a20ab1c3618e3ae7998652418be0d15d9763439`; size: `3028461877` bytes.
- Container `visiongaurd-api-1`: `running`, health `healthy`, restart count `0`. Docker engine was available.
- External `GET http://127.0.0.1:8000/health`: HTTP 200; `model_loaded: true`.
- External multipart `POST http://127.0.0.1:8000/predict` using `data\visa_prepared\1cls\candle\test\bad\000.JPG`: HTTP 200, image dimensions `1284x1168`, score `27.9863`, threshold `46.8800`, predicted `normal`. This individual thresholded miss is consistent with the measured low candle image recall and is not presented as a correct anomaly classification.
- All returned artifact URLs returned HTTP 200 with non-empty bodies: original `1037714` bytes, heatmap `348407`, overlay `1518360`, legend `419`.
- The live image build predates this evaluation-only code and documentation update; the API/Docker inputs were unchanged.

## Phase 2 controlled scoring experiment and final verification

Validation time: `2026-10-20T16:40:24+05:30`.

### Experiment and comparison

- Candidate configuration: `patchcore_topk_mean_1pct_v1`, mean of the largest 1% (8/784) patch distances for image scoring. Pixel maps and pixel threshold are unchanged. The baseline memory bank is reused.
- Training/fitting data: existing models fitted on official VisA `train/good`; the candidate image threshold was recalibrated with normal training images only at the existing 99th percentile. The official test split was used only for final measurements; no test labels/masks were used for fitting or calibration.
- Categories measured: candle, macaroni1, macaroni2. All three had 200 test images and zero missing masks.
- Three-category macro image AUROC/AP: `0.869033/0.855525` baseline and `0.877100/0.867851` candidate. Image precision/recall/F1: `0.852849/0.263333/0.375602` baseline and `0.915789/0.230000/0.359668` candidate.
- Candidate per-category image AUROC/AP/recall/F1: candle `0.9314/0.9311/0.17/0.2906`; macaroni1 `0.9335/0.9267/0.36/0.5217`; macaroni2 `0.7664/0.7458/0.16/0.2667`. Macaroni2 regressed; pixel metrics were identical by design.
- Decision: do not expand to the other nine categories and do not promote the candidate. Ranking metrics improved slightly at macro level, but macro thresholded recall and F1 fell, macaroni2 regressed, pixel localization did not change, and CPU latency increased. No performance improvement is claimed.
- Detailed measured tables and deltas are in `reports/phase2_comparison.json`, `reports/phase2_comparison.csv`, and the Phase 2 section of `reports/evaluation_report.md`. Test-split diagnostic visualizations and raw maps remain separated under the baseline and candidate experiment folders.

### Matched CPU benchmark

- Baseline command: `python -m scripts.benchmark --model artifacts\model.npz --dataset-root data\visa_prepared\1cls --category candle --warmup 20 --samples 100 --output reports\experiments\phase2_baseline_benchmark.json --run-name phase2-baseline-max-candle`
- Candidate command: `python -m scripts.benchmark --model artifacts\experiments\patchcore_topk_mean_1pct_v1\candle\model.npz --dataset-root data\visa_prepared\1cls --category candle --warmup 20 --samples 100 --output reports\experiments\patchcore_topk_mean_1pct_v1\benchmark_results.json`
- Both benchmarks used 20 warmups and 100 candle test images (224x224 model input), CPU, Intel(R) Core(TM) Ultra 7 270K Plus, Python 3.12.13, PyTorch 2.5.1+cpu, and NumPy 1.26.4. CUDA was unavailable.
- Baseline mean/median/p95 latency: `62.152/62.164/66.341 ms`; throughput `16.090 images/s`; model load `2.016 s`; peak working set `820477952` bytes.
- Candidate mean/median/p95 latency: `63.922/64.223/70.941 ms`; throughput `15.644 images/s`; model load `2.020 s`; peak working set `820277248` bytes.
- Baseline/candidate model artifact sizes: `258677895/258677966` bytes; memory-bank size `4337664` bytes each. The matched measurements have no confidence intervals.

### MLflow, tests, and deployment

- MLflow experiment `visionguard-visa`: 46/46 runs `FINISHED`. The distinct baseline and candidate benchmark runs each log dataset identity, category, CPU device, aggregation, 20 warmups, 100 samples, timing/throughput/working-set metrics, and a non-empty JSON benchmark artifact. The three candidate category runs retain evaluation/config/model and available diagnostic artifacts; historical runs were preserved.
- Full tests: `.\.venv\Scripts\python.exe -m pytest -q` — `43 passed, 0 failed` in `37.96 s`, with 18 dependency deprecation/user warnings (Starlette/AnyIO, MLflow `pkg_resources`, and MLflow gateway Pydantic v1 validators). An initial run had two API-test failures because the lightweight test fixture omitted the now-required config; the fixture was corrected to use default `ModelConfig`, and the full rerun passed.
- Compilation: `.\.venv\Scripts\python.exe -m compileall -q app visionguard scripts tests` — `PASS`.
- Docker Compose config: `docker compose config -q` — `PASS`. Existing container was not restarted or rebuilt; image `sha256:243310d44bc649ebb753a8c63a20ab1c3618e3ae7998652418be0d15d9763439`, size `3028461877` bytes, remains `running`/`healthy` with zero restarts and serves the baseline artifact.
- Live `GET /health`: HTTP 200, `model_loaded: true`. Real VisA candle anomaly `data\visa_prepared\1cls\candle\test\bad\000.JPG`: `/predict` HTTP 200, score `27.9863`, threshold `46.8800`, predicted `normal` (a measured false negative consistent with candle recall, not a successful detection).
- The prediction returned original, heatmap, overlay, and legend paths. Each relative artifact URL returned HTTP 200 with a non-empty body of `1037714`, `348407`, `1518360`, and `419` bytes, respectively. Recent container logs showed successful health and prediction requests and no inference error.
- The first external URL verification assumed absolute URLs; the API correctly returns relative `/artifacts/...` paths. The check was corrected to resolve those paths against `http://127.0.0.1:8000`, after which all four artifact checks passed.
- No Docker rebuild was needed: the candidate is not deployed, and the running service still loads the unchanged baseline model/artifact and retained default max-scoring behavior.

## Phase 3 small-defect localization experiment

### Diagnosis and protocol

- Verified the historical baseline from `reports/evaluation_results.json`: 12 measured categories; macro image AUROC/AP `0.88895979/0.89952682`, macro pixel AUROC/AP `0.95630217/0.31864184`, and image precision/recall/F1 `0.93579593/0.42083333/0.54915726`. The saved Phase 2 comparison contains candle, macaroni1, and macaroni2.
- A real 224x224 input produced feature shape `(784, 1536)` with spatial grid `(28, 28)`. Wide-ResNet50-2 progresses from conv1 112x112, maxpool/layer1 56x56 (layer1 has 256 channels), layer2 28x28 (512 channels), and layer3 14x14 (1024 channels). Layer3 is bilinearly aligned to layer2 before their 1536 channels are concatenated.
- `preprocess_image` converts to RGB, resizes the entire source image to a square 224x224 using PIL bilinear interpolation (aspect ratio is not preserved), then applies ImageNet channel normalization. The 28x28 patch-distance map is subsequently resized to source-image dimensions. Test masks are filename-matched; the evaluator aligns any differing dimensions with nearest-neighbor mask resizing. Existing full VisA validation checked the official anomalous mask pairs for equal dimensions.
- The candidate was fixed to change only raw map upsampling from bilinear to nearest. It retains baseline feature vectors, patch distances, train-only memory banks, image max scores and saved image thresholds. Pixel thresholds were calibrated only on normal official train/good maps at the 99.5th percentile. The weighted percentile exactly matches the flattened nearest-upsampled maps while avoiding retaining hundreds of full-resolution calibration maps. Official test labels/masks were used only for final metrics and diagnostics.
- An initial full-resolution calibration attempt was stopped after its worker reached approximately 4.2 GiB working set and 6.0 GiB private memory while processing normal maps. The bounded weighted calibration replaced that path. The exact pixel metric computation remains memory intensive: the macaroni2 evaluation worker was observed at 8.21 GiB working set during evaluation; this observation is not asserted as a process peak.

### Results and decision

- Categories: candle, macaroni1, macaroni2; 200 official test images per category and zero missing anomaly masks.
- Image-level metrics are unchanged for every category because image scoring uses the same 784 raw patch distances and the same saved threshold.
- Three-category macro pixel AUROC/AP/precision/recall/F1: baseline `0.96310891/0.09144838/0.03891415/0.70326280/0.07263556`; nearest candidate `0.93101597/0.07647605/0.03839190/0.61984939/0.07118515`.
- Pixel AP declined in every category: candle `0.14886 → 0.12527`, macaroni1 `0.08281 → 0.06258`, macaroni2 `0.04267 → 0.04158`. Pixel AUROC and recall also declined in every category. Pixel F1 was nearly unchanged in candle and slightly higher in macaroni1, but declined in macaroni2; this does not offset the AP/AUROC regressions.
- Decision: retain the baseline and do not expand nearest interpolation to the remaining nine categories. No localization improvement is claimed.
- Full per-category/macro results, thresholds, and diagnostic metadata are in `reports/experiments/phase3_localization_v1/comparison.json` and `.csv`. Baseline and candidate true-positive/false-positive/false-negative image, mask, heatmap, and overlay files are stored under each category's `examples/baseline/` and `examples/nearest/` folders. False-positive masks are explicit zero masks at original image dimensions.

### Runtime and benchmarks

- Existing baseline MLflow stage durations (fit/calibration/evaluation seconds), respectively: candle `69.126/134.382/280.710`; macaroni1 `67.197/130.530/267.868`; macaroni2 `61.791/129.829/269.579`.
- Candidate model fitting was not performed; existing train-only fitted banks/backbones were reused. Candidate calibration/evaluation seconds: candle `69.455/216.183`; macaroni1 `64.295/215.640`; macaroni2 `64.338/214.930`.
- Isolated CPU benchmarks used 20 warmups and 100 samples per model/category on the Intel Core Ultra 7 270K Plus, Python 3.12.13, PyTorch 2.5.1+cpu, NumPy 1.26.4. Baseline versus nearest mean/median/p95 milliseconds: candle `70.812/67.081/83.679` vs `68.657/64.792/83.025`; macaroni1 `69.009/65.797/83.736` vs `67.009/65.636/77.246`; macaroni2 `68.869/68.331/75.807` vs `65.333/65.396/70.247`.
- Per-run peak working sets were baseline/candidate `820457472/821035008` bytes for candle, `820817920/819630080` for macaroni1, and `822300672/820895744` for macaroni2. Inference benchmarks ran in separate processes. Timing runs have no confidence intervals; the modest measured speed difference is not treated as a localization gain.

### Final verification

- `.\.venv\Scripts\python.exe -m pytest -q`: `51 passed, 0 failed` in `53.54 s`, with 18 dependency deprecation/user warnings.
- `.\.venv\Scripts\python.exe -m compileall -q app visionguard scripts tests`: `PASS`.
- MLflow `visionguard-visa`: 55 runs total, all finished; nine uniquely named Phase 3 runs (three category evaluations and six separate CPU benchmark runs) contain dataset identity, configuration/parameters, measured metrics, and artifacts. The three candidate evaluation runs also contain model/config/report and diagnostic artifacts. No historical run was removed.
- `docker compose config -q`: `PASS`. The existing container was not rebuilt or restarted; it remains running/healthy with zero restarts on image `sha256:243310d44bc649ebb753a8c63a20ab1c3618e3ae7998652418be0d15d9763439` (3,028,461,877 bytes), and continues to load the baseline model.
- Live `/health`: HTTP 200, `model_loaded: true`. A real VisA candle anomaly prediction returned HTTP 200 with score `27.9863`, threshold `46.8800`, and status `normal` (a false negative, consistent with the baseline candle recall). Original, heatmap, overlay, and legend artifact paths each returned HTTP 200 with non-empty responses (`1037714`, `348407`, `1518360`, and `419` bytes).

## Phase 4 multiscale feature experiment

### Candidate and protocol

- Created isolated experiment `phase4_multiscale_v1`; the production model artifacts, reports from earlier phases, and running Docker deployment were not modified.
- Confirmed real feature shapes for a 224x224 input: layer1 `(1, 256, 56, 56)`, layer2 `(1, 512, 28, 28)`, and layer3 `(1, 1024, 14, 14)`. Layer2 and layer3 were bilinearly aligned to layer1's 56x56 grid. Each scale was projected to 128 seeded channels, normalized per location, equally weighted, and concatenated into 384-D descriptors (3,136 patches/image).
- Candidate memory banks were built from the official 900 `train/good` images/category only, using seed 42 and greedy farthest-point selection from the seeded 6,000-descriptor pool. Each bank is 2,823x384 (4,336,128 bytes), compared with baseline 706x1,536 (4,337,664 bytes).
- Candidate thresholds were independently calibrated on normal `train/good` only: exact 99th-percentile image score and a streamed 262,144-bin histogram approximation for the 99.5th-percentile pixel score. The official test split and its masks were reserved for final evaluation and diagnostics; no test labels or masks were used to select features or thresholds.
- Compared the baseline and fixed candidate on the same official test examples for candle, macaroni1, and macaroni2. Baseline values here are the measured three-category subset and should not be confused with the historical 12-category Phase 2 macro results.

### Results and promotion decision

| Category | Pixel AP baseline → candidate | Image recall baseline → candidate | Mean CPU latency baseline → candidate (ms) | Candidate evaluation peak working set |
|---|---:|---:|---:|---:|
| candle | 0.1489 → 0.1262 | 0.030 → 0.060 | 70.81 → 145.93 | 17,793,138,688 bytes |
| macaroni1 | 0.0828 → 0.0962 | 0.410 → 0.360 | 69.01 → 139.02 | 16,991,547,392 bytes |
| macaroni2 | 0.0427 → 0.0757 | 0.350 → 0.110 | 68.87 → 158.58 | 5,626,642,432 bytes |

- Three-category macro pixel AUROC/AP changed from `0.963109/0.091448` to `0.978511/0.099347`; pixel precision/recall/F1 changed from `0.038914/0.703263/0.072636` to `0.045869/0.702401/0.084642`.
- Three-category macro image AUROC/AP changed from `0.869033/0.855525` to `0.810600/0.779562`; image precision/recall/F1 changed from `0.852849/0.263333/0.375602` to `0.840302/0.176667/0.271923`.
- Localization AP improved on both macaroni categories but declined on candle. Image-level ranking declined in all three categories, and macaroni2 image recall dropped from 0.35 to 0.11. Candidate mean CPU latency was roughly twice the baseline; candidate artifacts were about 261.2 MB versus 258.7 MB. The benchmark working set increased by roughly 100 MB; measured evaluation working sets were much higher, with only 426 MB minimum available system memory during macaroni1 evaluation. Candle's saved evaluation peak predates the disk-backed ranking implementation, and macaroni1's saved JSON does not preserve an implementation marker; macaroni2 records the disk-backed sorter explicitly. These resource observations are reported as measured and are not treated as a promotion-quality result.
- **Decision: reject the multiscale candidate and retain the deployed baseline.** The modest macro pixel AP/AUROC gains do not offset candle's localization regression, poorer image detection and recall, slower inference, and evaluation memory risk. Results do not justify promotion or expansion beyond the three target categories.

### Diagnostics, artifacts, and verification

- Generated baseline/candidate true-positive, false-negative, and false-positive diagnostics. The prior baseline heatmap PNGs had width/height transposed; Phase 4 now regenerates baseline heatmaps and overlays from the saved raw anomaly maps at source dimensions and updates the copied metadata paths. All 18 diagnostic cases passed original/mask/heatmap/overlay dimension checks.
- Reports, configuration, category-specific models, thresholds, metrics, benchmarks, and diagnostics are under [`experiments/phase4_multiscale_v1`](experiments/phase4_multiscale_v1/). The report generator now resolves experiment-relative diagnostic paths correctly and logs an aggregate `phase4-multiscale-summary` run.
- MLflow experiment `visionguard-visa` contains 13 uniquely named Phase 4 runs (four stages per category plus the aggregate summary); all 13 are `FINISHED`. The summary run contains the comparison reports, configuration, benchmarks, and diagnostics.
- `.\.venv\Scripts\python.exe -m compileall -q app visionguard scripts tests`: `PASS`.
- `.\.venv\Scripts\python.exe -m pytest -q`: `61 passed, 0 failed` in `58.04 s`; 18 dependency/deprecation warnings.
- `docker compose config --quiet`: `PASS`. No rebuild or restart was performed. `visiongaurd-api-1` remains `Up` and `healthy`; `/health` returned `model_loaded: true`.
- A real candle test image (`014.JPG`, 1284x1168) returned score `55.9602`, threshold `46.8800`, and `anomalous`. All four response artifact URLs (original, heatmap, overlay, legend) returned HTTP 200 with non-empty bodies (`984373`, `210980`, `1264086`, and `419` bytes).
