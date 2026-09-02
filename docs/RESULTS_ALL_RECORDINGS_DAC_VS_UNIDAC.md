# All recordings: matched DAC versus UniDAC results

This report completes the matched metric-depth comparison on the four G1_A
fisheye sequences. In total, each model was evaluated on 1,204 frames and the
same 1,005 person detections. The physical-reference comparison contains 739
valid ZED detections and 725 stereo-gated LiDAR detections per model.

The `recording2`–`recording4` runs used project commit `2923c0e`, pinned DAC
commit `371ee29`, pinned UniDAC commit `9ddfc1f`, YOLO26n-seg 8.4.67, and
ByteTrack. Every available frame was processed with `frame_step = 1`; no depth
predictions were missing. Depth generation and the controlled timing benchmark
ran locally on Apple Metal (M1 Pro). The earlier matched `recording1` accuracy
result was produced on a Tesla T4. Accuracy is combined across hardware because
the saved metric predictions use the same evaluation protocol, but latency is
reported separately by hardware.

## Accuracy over all four recordings

Accuracy means are weighted by the number of valid physical-reference person
detections in each recording.

| Metric | DAC | UniDAC | UniDAC reduction |
| --- | ---: | ---: | ---: |
| Mean ZED common-pixel MAE | 0.776 m | 0.580 m | 25.2% |
| Mean ZED common-pixel RMSE | 1.122 m | 0.924 m | 17.6% |
| Mean ZED AbsRel | 0.376 | 0.233 | 37.9% |
| Mean ZED 3D localization error | 0.508 m | 0.348 m | 31.5% |
| Mean LiDAR 3D localization error | 0.475 m | 0.330 m | 30.5% |

UniDAC is the stronger overall accuracy model for this dataset. It reduces the
two primary 3D person-localization errors by about 31% relative to DAC.

## Remaining recordings run locally

| Recording | Model | Frames | Detections | ZED refs | LiDAR refs | ZED MAE | ZED 3D | LiDAR 3D |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recording2 | DAC | 124 | 86 | 65 | 63 | 0.603 m | 0.541 m | 0.542 m |
| recording2 | UniDAC | 124 | 86 | 65 | 63 | 0.292 m | 0.171 m | 0.191 m |
| recording3 | DAC | 791 | 698 | 470 | 461 | 0.767 m | 0.524 m | 0.486 m |
| recording3 | UniDAC | 791 | 698 | 470 | 461 | 0.542 m | 0.290 m | 0.249 m |
| recording4 | DAC | 156 | 123 | 123 | 123 | 0.958 m | 0.374 m | 0.321 m |
| recording4 | UniDAC | 156 | 123 | 123 | 123 | 0.983 m | 0.693 m | 0.717 m |
| **recordings2–4 weighted** | **DAC** | **1,071** | **907** | **658** | **647** | **0.787 m** | **0.498 m** | **0.460 m** |
| **recordings2–4 weighted** | **UniDAC** | **1,071** | **907** | **658** | **647** | **0.600 m** | **0.353 m** | **0.332 m** |

The exact machine-readable table is available in
[`data/recordings2_to_4_dac_vs_unidac.csv`](data/recordings2_to_4_dac_vs_unidac.csv).

## Robustness and the recording4 failure segment

The pooled `recording2`–`recording4` medians support the aggregate conclusion:

| Metric | DAC median | UniDAC median |
| --- | ---: | ---: |
| ZED 3D localization error | 0.475 m | 0.216 m |
| LiDAR 3D localization error | 0.460 m | 0.221 m |

The model ranking is not uniform. UniDAC is clearly better on `recording2` and
`recording3`, while DAC has the lower *mean* error on `recording4`. That reversal
is caused by a localized far-range failure rather than uniformly worse UniDAC
predictions: its `recording4` median LiDAR error is 0.164 m, lower than DAC's
0.298 m, but 24 UniDAC frames exceed 1 m LiDAR error. In frames 102–123,
UniDAC predicts the person at roughly 8.4–10.4 m while LiDAR places the person
at roughly 6.1–7.0 m. UniDAC's LiDAR-error P90 is therefore 2.885 m, compared
with 0.595 m for DAC.

This segment must be shown in the final report. It demonstrates that the
overall winner can still have a safety-relevant failure mode at longer range.

## Localization error by LiDAR reference distance

The plot below uses the 647 valid person/LiDAR matches per model from
`recording2` to `recording4`. Each translucent point is one detection. The
solid lines show median 3D localization error in occupied 1 m reference-distance
bins. UniDAC has lower median error through most of the measured range, while
the sharp increase beyond 6 m exposes its localized `recording4` failure.

![3D person localization error versus LiDAR reference distance](images/person_localization_error_vs_lidar_distance.png)

## Controlled Apple Metal speed result

The dedicated benchmark used ten fixed `recording1` frames, five warm-up runs
per model, and ten timed runs per frame and model. Model loading, image loading,
person detection, tracking, and disk output were excluded.

| Model | Measurements | Mean | Median | P10 | P90 | FPS from median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DAC | 100 | 390.44 ms | 390.35 ms | 387.76 ms | 392.91 ms | 2.562 |
| UniDAC | 100 | 1088.78 ms | 1086.51 ms | 1075.72 ms | 1104.64 ms | 0.920 |

DAC is 2.783 times faster by median latency on the M1 Pro. These numbers must
not be combined with the earlier Tesla T4 timing values.

## Recommendation

Use UniDAC as the accuracy-oriented metric-depth baseline and DAC as the
speed-oriented baseline. Report the roughly 31% overall 3D-error reduction
alongside the 2.8-times local latency cost and the `recording4` far-range
failure. Before a safety claim, investigate that failure segment and evaluate
more long-range sequences; the current result supports model selection for the
semester experiment, not deployment certification.
