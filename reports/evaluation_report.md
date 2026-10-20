# Evaluation report

- Status: `measured`
- Dataset type: `VisA`
- Fitting/calibration split: `train/good` only
- Final evaluation split: `test`
- Thresholds: 99th/99.5th percentile of train/good normal scores
- Image detection and pixel localization are reported separately.
- Pixel metrics are null when any anomalous test image lacks its ground-truth mask.

| Category | Image AUROC | Image AP | Image F1 | Pixel AUROC | Pixel AP | Pixel F1 | Test images | Missing masks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candle | 0.8718999999999999 | 0.8170888315495455 | 0.057692307692307696 | 0.9672104181324164 | 0.14886078120733756 | 0.1313760929223863 | 200 | 0 |
| capsules | 0.7578333333333334 | 0.8645117513160071 | 0.4126984126984127 | 0.9500742568869096 | 0.43992500005440327 | 0.3412311934129494 | 160 | 0 |
| cashew | 0.9126 | 0.9567578303868756 | 0.5972222222222222 | 0.9360722404510866 | 0.4212816092994832 | 0.45204536183908683 | 150 | 0 |
| chewinggum | 0.9880000000000001 | 0.9943724940703498 | 0.9368421052631579 | 0.9811177328210133 | 0.5042014047387671 | 0.3179322761702068 | 150 | 0 |
| fryum | 0.9354 | 0.9708157657344616 | 0.7878787878787878 | 0.9015083176230145 | 0.28807136567549246 | 0.36878886398208544 | 150 | 0 |
| macaroni1 | 0.9184 | 0.9152829869757733 | 0.5655172413793104 | 0.9728629068457302 | 0.08281085781166449 | 0.045771178316670616 | 200 | 0 |
| macaroni2 | 0.8168 | 0.8342045146286579 | 0.5035971223021583 | 0.949253406095515 | 0.0426735022859332 | 0.0407593940189142 | 200 | 0 |
| pcb1 | 0.8586 | 0.8138390047832051 | 0.417910447761194 | 0.981813016099461 | 0.5931143473617541 | 0.3360867433600107 | 200 | 0 |
| pcb2 | 0.8180000000000001 | 0.8190362086262534 | 0.32786885245901637 | 0.9421059786106266 | 0.1675938449212748 | 0.15527717528913873 | 200 | 0 |
| pcb3 | 0.8574257425742574 | 0.8693951677687632 | 0.5147058823529411 | 0.9729854789579865 | 0.3273023889297995 | 0.1785377701201918 | 201 | 0 |
| pcb4 | 0.9441584158415841 | 0.9448074716870141 | 0.5915492957746479 | 0.9490883753594396 | 0.34821940822248604 | 0.2899118675359943 | 201 | 0 |
| pipe_fryum | 0.9884000000000001 | 0.9942097770348398 | 0.8764044943820225 | 0.9715339128322461 | 0.45964753111805945 | 0.45967883482152316 | 150 | 0 |

## Macro-average

Equal-weight arithmetic mean across categories with a defined metric:

```json
{
  "image_auroc": 0.8889597909790979,
  "image_average_precision": 0.8995268170468123,
  "image_precision": 0.9357959306488718,
  "image_recall": 0.4208333333333334,
  "image_f1": 0.5491572643471816,
  "image_false_positive_rate": 0.018325082508250826,
  "image_false_negative_rate": 0.5791666666666666,
  "pixel_auroc": 0.9563021700596205,
  "pixel_average_precision": 0.3186418368022046,
  "pixel_precision": 0.17659506950679485,
  "pixel_recall": 0.7304103782919152,
  "pixel_f1": 0.2597830626490965,
  "pixel_false_positive_rate": 0.01576434177120318,
  "pixel_false_negative_rate": 0.26958962170808465
}
```

## Failed categories

None.

## Pending categories

None.

## Phase 2 diagnostic findings

- The baseline uses the maximum patch distance for image scoring. Its image
  threshold is the 99th percentile of normal `train/good` scores, not a
  validation-optimized operating point. This normal-only rule limits false
  alarms but does not guarantee high defect recall; candle recall is 0.03 at
  its saved threshold.
- Image and mask pairs were matched by filename and dimensions in the VisA
  evaluation. No evidence of a systematic mask-alignment or metric-definition
  error was found.
- Pixel AP is sensitive to the very sparse defect pixels in macaroni1 and
  macaroni2 (about 0.036% and 0.029% foreground pixels in the inspected
  masks), even where pixel AUROC is high. This class imbalance is consistent
  with, but does not by itself prove the cause of, the low AP.
- The 28x28 feature grid is resized to full image dimensions for localization.
  Small defects can therefore be spatially coarse or diluted. The current
  results do not establish how much this resolution contributes; no pixel-map
  change was tested in this experiment.
- The candidate changed image-score aggregation only. It left pixel maps and
  pixel metrics unchanged, while candle improved and macaroni2 regressed.
  The three-category macro recall/F1 and matched CPU latency worsened, so
  there is no evidence to promote or expand it.

## Phase 2 controlled experiment

- Experiment: `patchcore_topk_mean_1pct_v1`
- Candidate: mean of the highest 1% of spatial patch distances (8 of 784 patches at 224x224).
- The fitted baseline memory bank is reused. Candidate image thresholds use only normal `train/good` scores at the 99th percentile; the pixel map and baseline normal-only pixel threshold are unchanged.
- Test labels/masks are used only for these final measurements, not for choosing the candidate or calibrating thresholds.

| Category | Image AUROC baseline → candidate | Image AP baseline → candidate | Image recall baseline → candidate | Image F1 baseline → candidate | Pixel AUROC baseline → candidate | Pixel AP baseline → candidate |
|---|---:|---:|---:|---:|---:|---:|
| candle | 0.8719 → 0.9314 | 0.8171 → 0.9311 | 0.0300 → 0.1700 | 0.0577 → 0.2906 | 0.9672 → 0.9672 | 0.1489 → 0.1489 |
| macaroni1 | 0.9184 → 0.9335 | 0.9153 → 0.9267 | 0.4100 → 0.3600 | 0.5655 → 0.5217 | 0.9729 → 0.9729 | 0.0828 → 0.0828 |
| macaroni2 | 0.8168 → 0.7664 | 0.8342 → 0.7458 | 0.3500 → 0.1600 | 0.5036 → 0.2667 | 0.9493 → 0.9493 | 0.0427 → 0.0427 |

### Three-category macro average

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| image_auroc | 0.869033 | 0.877100 | +0.008067 |
| image_average_precision | 0.855525 | 0.867851 | +0.012326 |
| image_precision | 0.852849 | 0.915789 | +0.062940 |
| image_recall | 0.263333 | 0.230000 | -0.033333 |
| image_f1 | 0.375602 | 0.359668 | -0.015934 |
| pixel_auroc | 0.963109 | 0.963109 | +0.000000 |
| pixel_average_precision | 0.091448 | 0.091448 | +0.000000 |
| pixel_precision | 0.038914 | 0.038914 | +0.000000 |
| pixel_recall | 0.703263 | 0.703263 | +0.000000 |
| pixel_f1 | 0.072636 | 0.072636 | +0.000000 |
| inference_seconds_per_image | 0.073109 | 0.078933 | +0.005825 |

### CPU benchmark comparison

- Device: `cpu` on `Intel(R) Core(TM) Ultra 7 270K Plus`; both runs used 20 warmups and 100 measured candle test images at 224x224 input. Candidate ran on the same hardware/software configuration.
- Software (baseline): Python 3.12.13, PyTorch 2.5.1+cpu, NumPy 1.26.4.

| Metric | Baseline max | Candidate top-1% mean |
|---|---:|---:|
| Model load (s) | 2.016 | 2.020 |
| Mean prediction latency (ms) | 62.152 | 63.922 |
| Median prediction latency (ms) | 62.164 | 64.223 |
| P95 prediction latency (ms) | 66.341 | 70.941 |
| Throughput (images/s) | 16.090 | 15.644 |
| Peak process working set (bytes) | 820477952 | 820277248 |
| Model artifact size (bytes) | 258677895 | 258677966 |
| Memory bank size (bytes) | 4337664 | 4337664 |

Machine-readable comparison: `reports\phase2_comparison.json` and `reports\phase2_comparison.csv`.
Diagnostic examples are in `reports/experiments/baseline_patchcore/` and the candidate experiment's `examples/` folders. True positives, false positives, and false negatives are labeled by the thresholded prediction; a false negative is not a detection.

## Phase 3 localization experiment

- Experiment: `phase3_localization_v1`, evaluated on candle, macaroni1, and macaroni2.
- The only candidate change is nearest-neighbor resizing of the existing 28x28 raw patch-distance map, instead of baseline bilinear interpolation. The Wide-ResNet50-2 feature extractor, patch distances, 1536-dimensional feature vectors, 784-patch grid, memory bank, image score, preprocessing, data split, and metric definitions remain unchanged.
- Candidate memory banks are the existing train-only fitted banks. The unchanged baseline image threshold is used. Candidate pixel thresholds are exactly calibrated at the 99.5th percentile from normal `train/good` patch maps weighted by nearest-resize pixel coverage; test labels/masks are only used for final evaluation.

| Category | Pixel AUROC baseline → nearest | Pixel AP baseline → nearest | Pixel precision baseline → nearest | Pixel recall baseline → nearest | Pixel F1 baseline → nearest |
|---|---:|---:|---:|---:|---:|
| candle | 0.9672 → 0.9480 | 0.1489 → 0.1253 | 0.0720 → 0.0720 | 0.7442 → 0.6836 | 0.1314 → 0.1303 |
| macaroni1 | 0.9729 → 0.9397 | 0.0828 → 0.0626 | 0.0236 → 0.0240 | 0.7877 → 0.6943 | 0.0458 → 0.0465 |
| macaroni2 | 0.9493 → 0.9053 | 0.0427 → 0.0416 | 0.0211 → 0.0191 | 0.5779 → 0.4816 | 0.0408 → 0.0368 |

Macro image AUROC/AP/precision/recall/F1 are unchanged at `0.8690/0.8555/0.8528/0.2633/0.3756`. Macro pixel AUROC/AP/precision/recall/F1 changed from `0.9631/0.09145/0.03891/0.7033/0.07264` to `0.9310/0.07648/0.03839/0.6198/0.07119`. Therefore, retain the baseline and do not expand the candidate to the other nine categories.

The candidate was 2.15–3.54 ms faster in the per-category 100-sample CPU runs, with peak inference-benchmark working sets around 0.82 GB for both configurations. This latency result does not counter the localization regressions. Per-category benchmark statistics, stage durations, thresholds, full metrics, and baseline/candidate true-positive, false-positive, and false-negative diagnostic images are in `reports/experiments/phase3_localization_v1/`.
