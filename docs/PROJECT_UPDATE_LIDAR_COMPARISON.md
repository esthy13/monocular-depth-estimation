# Project update: DAC and UniDAC comparison with LiDAR

**Date:** 19 August 2026  
**Suggested subject:** Project update: DAC and UniDAC comparison with LiDAR

Dear Christopher,

We wanted to send you a short update on our results for 3D person localization
with the G1_A fisheye camera. We compared DAC and UniDAC on the same RGB frames,
person detections, tracks, camera calibration, and physical reference data.

Across the four recordings, each model processed 1,204 frames with the same
1,005 person detections. For 725 of those detections, we had a valid LiDAR
reference. We matched the measurements by timestamp, transformed them into the
G1_A coordinate system, and used stereo depth to reduce errors caused by the
different sensor viewpoints.

| Recording | DAC mean LiDAR 3D error | UniDAC mean LiDAR 3D error |
| --- | ---: | ---: |
| Recording 1 | 0.595 m | 0.312 m |
| Recording 2 | 0.542 m | 0.191 m |
| Recording 3 | 0.486 m | 0.249 m |
| Recording 4 | 0.321 m | 0.717 m |
| **All recordings, weighted** | **0.475 m** | **0.330 m** |

Overall, UniDAC reduced the mean LiDAR 3D localization error by **30.5%**
compared with DAC. It performed better in recordings 1 to 3. Recording 4 was
the exception. In 24 frames, UniDAC had more than 1 m error and estimated the
person at about 8.4 to 10.4 m, while LiDAR measured about 6.1 to 7.0 m. We plan
to investigate this far range failure and include it clearly in the report.

There is also a clear speed trade-off. In our controlled M1 Pro benchmark, the
median depth time was 390 ms for DAC, or 2.56 FPS, and 1,087 ms for UniDAC, or
0.92 FPS. DAC was therefore about 2.8 times faster.

We have not included Depth Anything V2 in the metric ranking because its output
is relative rather than metric. Fitting its scale and shift on the same LiDAR
points is useful for checking the predicted depth shape, but it would not be a
fair test of metric accuracy. For that comparison, we would need to calibrate
it on separate frames and evaluate it on held-out frames.

At this stage, our conclusion is to use UniDAC as the accuracy-focused model and
DAC as the faster baseline. We are treating LiDAR as a physical reference rather
than perfect ground truth because calibration, synchronization, and viewpoint
differences can also contribute to the measured error.

We would be glad to hear whether you think this evaluation and our planned
follow-up on the long-range failure cover the main points needed for the final
report.

Best regards,  
Esther Giuliano and Fazel Malekian

Detailed results are available in the
[all-recording DAC and UniDAC evaluation](RESULTS_ALL_RECORDINGS_DAC_VS_UNIDAC.md).
