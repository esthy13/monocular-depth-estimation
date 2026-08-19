# Recording 1: matched DAC versus UniDAC results

This comparison uses all 133 consecutive G1_A frames from `recording1`. Both
models were evaluated with the same RGB frames, person detections, tracks,
fisheye calibration, ZED references, LiDAR references, and evaluation code on a
Tesla T4. The runs contain the same 98 person detections, 81 valid ZED
references, and 78 valid LiDAR references, so their accuracy measurements are
directly comparable.

The UniDAC run used project commit `798a451` and pinned UniDAC commit `9ddfc1f`.
The DAC run used project commit `688ced5`, pinned DAC commit `371ee29`, and the
`dac-indoor-resnet101` checkpoint.

## Main result

| Metric | UniDAC | DAC | UniDAC reduction vs DAC |
|---|---:|---:|---:|
| Mean ZED common-pixel MAE | 0.419 m | 0.686 m | 39.0% |
| Mean ZED common-pixel RMSE | 0.600 m | 0.801 m | 25.1% |
| Mean ZED AbsRel | 0.422 | 0.870 | 51.5% |
| Mean ZED 3D localization error | 0.309 m | 0.594 m | 48.0% |
| Mean LiDAR 3D localization error | 0.312 m | 0.595 m | 47.6% |

The medians support the same conclusion and show that the improvement is not
caused only by a few outliers:

| Metric | UniDAC median | DAC median | UniDAC reduction vs DAC |
|---|---:|---:|---:|
| ZED common-pixel MAE | 0.335 m | 0.557 m | 39.9% |
| ZED common-pixel RMSE | 0.406 m | 0.572 m | 29.0% |
| ZED AbsRel | 0.427 | 0.692 | 38.3% |
| ZED 3D localization error | 0.306 m | 0.577 m | 46.9% |
| LiDAR 3D localization error | 0.328 m | 0.574 m | 42.8% |

Both models predict the person farther away than the physical references.
UniDAC has the smaller mean signed distance bias: +0.267 m versus ZED and
+0.277 m versus LiDAR, compared with DAC's +0.562 m and +0.578 m.

The difference is especially visible in the closer second track. Against ZED,
its mean AbsRel is 0.664 for UniDAC and 1.618 for DAC. This matters for the
project's person-safety use case, where close-range distance errors are
important.

## Speed trade-off

| Timing statistic | UniDAC | DAC |
|---|---:|---:|
| Mean depth time, all frames | 1270.5 ms | 862.7 ms |
| Median depth time, excluding first frame | 1215.3 ms | 807.0 ms |
| Approximate steady throughput | 0.823 FPS | 1.239 FPS |

DAC is about 1.5 times faster by the steady median measurement. These timings
contain only one timed inference per frame. A controlled benchmark with equal
warm-up and repeated runs is still required for the final inference-speed
claim.

## Tracking and reference coverage

- Both runs process 133/133 frames and produce the same 98 detections.
- Track 1 covers frames 0–60, frame 96 is detected before ID assignment, and
  track 3 covers frames 97–132.
- Both runs have exactly the same missing-reference frames. Most missing
  references occur near the fisheye boundary, where the physical sensors do not
  share the full G1_A field of view.
- The model choice therefore does not change detection, tracking, or reference
  coverage in this comparison; it changes depth accuracy and speed.

## Conclusion

For `recording1`, UniDAC is the better metric-depth model when 3D person
localization accuracy is the priority: it approximately halves the mean error
against both ZED and LiDAR. DAC remains the faster option, at roughly 1.5 times
the throughput. The current project baseline should therefore use UniDAC for
accuracy experiments and keep DAC as the speed-oriented comparison baseline.

This conclusion is limited to one recorded sequence, one camera, and one GPU.
It should be confirmed on the remaining recordings and with a controlled timing
benchmark before making a general model recommendation.
