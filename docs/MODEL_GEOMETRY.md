# Camera geometry used by each model

This note answers whether each model receives the calibrated camera intrinsics
or relies only on appearance learned during training.

| Component | Receives camera intrinsics? | Geometry behavior | Output |
|---|---:|---|---|
| Depth Anything V2 | No | Processes the image as a conventional image and relies on learned image priors; it has no explicit G1_A fisheye lens model | Relative inverse depth in arbitrary units |
| Depth Any Camera (DAC) | Yes | Uses the camera model, focal lengths, principal point, and fisheye distortion to project the image to an equirectangular representation and back | Metric depth in metres |
| UniDAC | Yes | Uses the same calibrated camera-to-ERP and ERP-to-camera geometry, including the widened crop required for the full circular G1_A field of view | Metric Euclidean ray distance in metres |
| YOLO26 + ByteTrack | No | Detects, segments, and associates people in image coordinates | Masks, boxes, and track IDs |
| 3D localization/evaluation | Yes | Uses G1_A intrinsics for ray back-projection and the calibrated extrinsics for ZED_B and E1_A/E1_B reprojection | G1_A camera-frame XYZ and physical-sensor errors |

## Calibration fields

The project reads the following fields from `intrinsic.json`:

- perspective camera: `fx`, `fy`, `cx`, and `cy` from the camera matrix;
- fisheye camera: the same camera matrix plus OpenCV fisheye coefficients
  `k1–k4`;
- camera model identifier: perspective or fisheye.

The 4×4 matrices in `extrinsics.json` transform ZED and LiDAR points into the
G1_A reference camera frame. The evaluator then projects those points with the
G1_A fisheye model.

## Depth convention

DAC and UniDAC outputs are metric depth, where smaller values are closer.
UniDAC metadata explicitly records `metric Euclidean ray distance`; the 3D
point at image pixel `(u, v)` is therefore:

```text
P_G1 = unit_fisheye_ray(u, v, K, distortion) × depth_metres
```

ZED_B provides perspective optical-axis depth. Before comparison, the evaluator
back-projects it to 3D in the ZED frame, transforms it into G1_A, and converts
it to Euclidean range. This prevents mixing incompatible Z-depth and ray-range
definitions.

Depth Anything V2 is not metric. Its numeric values cannot be inserted into the
3D equation or compared directly with ZED/LiDAR metres without a separate scale
and shift calibration.

## Report wording

It is accurate to say that Depth Anything V2 does not explicitly know the lens
calibration, whereas DAC and UniDAC receive and use it. Avoid saying that Depth
Anything V2 estimates the intrinsics: this project does not run an intrinsic
estimation stage for that model.
