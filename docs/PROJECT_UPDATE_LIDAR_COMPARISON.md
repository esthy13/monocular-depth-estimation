# Project update: DAC and UniDAC comparison with LiDAR

**Date:** 19 August 2026  
**Suggested subject:** Project update: DAC and UniDAC comparison with LiDAR

Dear Christopher,

Here is a short update on our 3D person localization results with the G1_A
fisheye camera. DAC and UniDAC both receive the calibrated fisheye intrinsics
and distortion parameters and return metric distance. We use YOLO26 for person
masks, ByteTrack for persistent IDs, and the median depth inside each person
mask to calculate a 3D point in the G1_A camera coordinate system.

We evaluated both models on the same 1,204 frames and 1,005 person detections
from four recordings. There were 725 valid LiDAR references per model. We
matched the sensors by timestamp and used stereo depth to reduce errors caused
by their different viewpoints.

The detection and tracking results were identical for both models, so the
comparison isolates the depth estimation. In recording 1, there were no ID
changes while the person remained visible. The person had one ID in frames 0
to 60 and a new ID in frames 97 to 132 after a long absence, which is expected
re-entry behavior.

| Recording | DAC mean LiDAR 3D error | UniDAC mean LiDAR 3D error |
| --- | ---: | ---: |
| Recording 1 | 0.595 m | 0.312 m |
| Recording 2 | 0.542 m | 0.191 m |
| Recording 3 | 0.486 m | 0.249 m |
| Recording 4 | 0.321 m | 0.717 m |
| **All recordings, weighted** | **0.475 m** | **0.330 m** |

Overall, UniDAC reduced the mean LiDAR 3D localization error by **30.5%**
compared with DAC. It performed better in recordings 1 to 3. Recording 4 was
the exception. In 24 frames, UniDAC had more than 1 m error and placed the
person at about 8.4 to 10.4 m, while LiDAR measured 6.1 to 7.0 m. We will
investigate and report this far range failure.

In our controlled M1 Pro benchmark, DAC took 390 ms per frame, or 2.56 FPS,
while UniDAC took 1,087 ms, or 0.92 FPS. DAC was about 2.8 times faster.

We have not included Depth Anything V2 in the metric ranking because its output
is relative. Fitting it on the same LiDAR points can check the predicted depth
shape, but it is not a fair metric test. A fair comparison would calibrate it on
separate frames and evaluate it on held-out frames.

Our current conclusion is to use UniDAC as the accuracy-focused model and DAC
as the faster baseline. We treat LiDAR as a physical reference rather than
perfect ground truth because calibration, synchronization, and viewpoint
differences can also affect the error. We would be glad to hear whether this
evaluation and our planned long-range follow-up cover the main points needed
for the final report.

Best regards,  
Esther Giuliano and Fazel Malekian

Detailed results are available in the
[all-recording DAC and UniDAC evaluation](RESULTS_ALL_RECORDINGS_DAC_VS_UNIDAC.md).
