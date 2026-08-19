# Recording 1: UniDAC full-sequence results

These results were produced from all 133 consecutive G1_A frames using project
commit `798a451` and pinned UniDAC commit `9ddfc1f`. Depth inference ran on a
Tesla T4. Person segmentation and tracking used YOLO26n-seg and ByteTrack.

## Completeness and tracking

- 133/133 depth frames were present and evaluated.
- 98 person detections were produced.
- Track 1 covers every frame from 0 through 60: 61 consecutive detections over
  5.99 seconds.
- Frame 96 is detected before ByteTrack assigns an ID.
- Track 3 covers every frame from 97 through 132: 36 consecutive detections
  over 3.50 seconds.
- There are no ID changes inside either continuously visible segment. The new
  ID after the long absence is expected re-entry behavior, not an in-view ID
  switch.

## Physical-sensor coverage

- 81/98 detections (82.7%) have a valid reprojected ZED reference.
- 78/98 detections (79.6%) have a stereo-gated LiDAR reference.
- Most missing reference measurements occur when the person is close to the
  fisheye periphery, where physical-sensor fields of view do not overlap.

## Accuracy

| Metric | Mean | Median | 90th percentile |
|---|---:|---:|---:|
| ZED common-pixel MAE | 0.419 m | 0.335 m | 0.732 m |
| ZED common-pixel RMSE | 0.600 m | 0.406 m | 1.320 m |
| ZED AbsRel | 0.422 | 0.427 | 0.686 |
| ZED 3D localization error | 0.309 m | 0.306 m | 0.410 m |
| LiDAR 3D localization error | 0.312 m | 0.328 m | 0.411 m |

UniDAC's median person distance is systematically farther than the references:
the mean signed median difference is +0.267 m versus ZED and +0.277 m versus
LiDAR. The error is proportionally larger during the close second track, where
the reference distance is often around 0.5 m.

Outlier frames require visual inspection before attributing the complete error
to UniDAC. Cross-camera occlusion and incomplete reference overlap can increase
common-pixel error, especially near the fisheye boundary.

## Speed

- Median UniDAC end-to-end depth time: 1.216 s/frame.
- Median excluding the initialization-heavy first frame: 1.215 s/frame.
- Approximate steady throughput: 0.823 FPS.
- Mean YOLO inference time on the Colab GPU: 16.3 ms/frame.

The depth timings contain one timed run per frame. The final speed comparison
must use the same GPU with controlled warm-up and repeated timed runs for both
UniDAC and DAC.

## Interpretation and matched baseline

The sequence demonstrates stable detection and ID continuity while a person is
continuously visible, but UniDAC has roughly 0.3 m 3D localization error against
both physical references and a positive distance bias. The subsequent matched
DAC run used the identical 133 frames and evaluation protocol. UniDAC reduced
the mean ZED and LiDAR 3D localization errors by 48.0% and 47.6%, respectively,
while DAC was about 1.5 times faster. See the
[matched comparison](RESULTS_RECORDING1_DAC_VS_UNIDAC.md) for the full result
and its limitations.
