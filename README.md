# monocular-depth-estimation

This research repository investigates whether a single RGB camera can support
3D person detection and tracking when monocular depth replaces a dedicated
depth sensor. It predicts depth for the calibrated G1_A fisheye and ZED_B
perspective cameras, localizes tracked people in 3D, and evaluates the estimates
against synchronized ZED stereo depth and two LiDAR sensors.

The implemented pipeline is:

1. generate relative or metric monocular depth from an RGB frame;
2. segment and track people with YOLO26 and ByteTrack;
3. back-project each person into the calibrated G1_A coordinate system;
4. compare the estimate with timestamp-matched ZED and LiDAR measurements; and
5. report accuracy, failure cases, and hardware-specific inference speed.

| Model | Output | Camera geometry | Role in this project |
| --- | --- | --- | --- |
| Depth Anything V2 | Relative depth | Inferred from image content; no camera intrinsics | Perspective visual baseline and oracle-aligned diagnostic |
| DAC | Metric depth (m) | Uses calibrated camera intrinsics and lens model | Faster fisheye metric baseline |
| UniDAC | Metric depth (m) | Uses calibrated camera intrinsics and lens model | Accuracy-oriented fisheye metric baseline |

This is experimental research code, not a safety-certified perception system.
The current result and known long-range failure are summarized below.

## Documentation

- [Experiment and execution guide](docs/EXPERIMENT_GUIDE.md)
- [Depth visualization scale policy](docs/VISUALIZATION_SCALE.md)
- [Camera geometry used by each model](docs/MODEL_GEOMETRY.md)
- [Recording 1 UniDAC full-sequence results](docs/RESULTS_RECORDING1_UNIDAC.md)
- [Recording 1 matched DAC versus UniDAC results](docs/RESULTS_RECORDING1_DAC_VS_UNIDAC.md)
- [All-recording matched DAC versus UniDAC results](docs/RESULTS_ALL_RECORDINGS_DAC_VS_UNIDAC.md)

## Repository structure

```md
monocular-depth-estimation/
├── docs/                       # experiment and geometry documentation
├── notebooks/                  # Colab GPU workflow
├── src/                        # models, geometry, tracking, and evaluation
├── tests/                      # regression tests
├── minutes/
├── references/
├── run_depth.py                # single-frame depth inference
├── run_metric_depth_batch.py   # resumable CUDA/MPS metric-depth batch
├── evaluate_person_tracking.py # 3D person/sensor evaluation
├── plot_evaluation_diagnostics.py # signed residual and distance-error plots
├── compare_evaluation_runs.py  # UniDAC/DAC accuracy report table
├── compare_evaluation_suite.py # weighted multi-recording comparison
├── summarize_lidar_suite.py    # validated perspective/LiDAR aggregation
├── analyze_lidar_outliers.py   # frame ranking and person-blur audit
├── project_lidar_fisheye.py    # calibrated LiDAR-on-G1_A projection
└── benchmark_depth_models.py   # controlled repeated GPU benchmark
```

## Local setup

Install Python 3.12 and [`uv`](https://docs.astral.sh/uv/), then create the
locked environment:

```bash
uv sync --locked
```

Run commands with `uv run python ...`; manual activation is optional.

The expected directory layout is:

```text
cv_project_data/
├── intrinsic.json
├── extrinsics.json
├── recording1/ ... recording4/
└── monocular-depth-estimation/   # this repository
```

Depth Anything V2 needs no additional source checkout. To run DAC or UniDAC
locally, clone the exact upstream revisions used by the completed experiments:

```bash
git clone https://github.com/yuliangguo/depth_any_camera.git third_party/depth_any_camera
git -C third_party/depth_any_camera checkout 371ee299429257bb9a27d1e23b7dc53670e37023

git clone https://github.com/girish1511/UniDAC.git third_party/UniDAC
git -C third_party/UniDAC checkout 9ddfc1f4cea68e08273ec9bca037f2ef9e1aa90e
```

Model checkpoints are downloaded from Hugging Face on first use and cached.
Before a long run, verify the command interfaces and project tests:

```bash
uv run python run_depth.py --help
PYTHONPATH=. uv run --with pytest pytest -q tests
```

### Local Apple Silicon execution

The full metric-depth batch can run from the VS Code terminal on Apple Metal.
`PYTORCH_ENABLE_MPS_FALLBACK=1` lets an unsupported operation fall back to the
CPU while the supported model layers remain on the M1/M2/M3 GPU. For example:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python run_metric_depth_batch.py \
  --data_dir .. \
  --output_dir ../local_experiments/unidac \
  --model UniDAC \
  --recordings recording1 recording2 recording3 recording4 \
  --device mps
```

Replace `UniDAC` and the output folder with `DAC` and `dac` for the matched
baseline. The command writes raw metric depth and metadata after every frame;
rerunning it skips completed pairs, so it can resume after sleep or interruption.
Keep the Mac connected to power and prevent system sleep during a long batch.

## Google Colab: matched DAC and UniDAC runs

Use the UniDAC notebook for the newer metric-depth candidate and the DAC
notebook for the established `dac-indoor-resnet101` baseline. Both notebooks
use the same calibrated output and evaluation protocol.

- UniDAC: [![Open UniDAC in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/esthy13/monocular-depth-estimation/blob/main/notebooks/unidac_colab.ipynb)
- DAC baseline: [![Open DAC in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/esthy13/monocular-depth-estimation/blob/main/notebooks/dac_colab.ipynb)
- Controlled speed benchmark: [![Open benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/esthy13/monocular-depth-estimation/blob/main/notebooks/benchmark_colab.ipynb)

Because this repository is private, an unauthenticated Colab link returns 404.
Open the notebook from Colab's GitHub picker after authorizing private-repository
access, or download the selected notebook while signed in to GitHub and choose
**File → Upload notebook** in Colab.

1. [Create a GitHub token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
   that can read this repository. Prefer a fine-grained, expiring token with
   read-only **Contents** permission when available.
2. In Colab's **Secrets** panel (key icon), add it as `GITHUB_TOKEN` and enable
   notebook access. Never paste the token into a code cell.
3. Upload `cv_project_data` to Google Drive.
4. Select **Runtime → Change runtime type → GPU**.
5. Run all cells. The first dependency setup restarts the Python kernel once;
   after Colab reconnects, select **Runtime → Run all** again.
6. Edit `DATA_DIR` and `OUTPUT_DIR` in the notebook's settings cell.
7. Run the one-frame smoke test before enabling its resumable batch cell.

Each notebook pins its official upstream source, caches its model checkpoint in
Drive, uses the calibrated G1_A/ZED_B camera geometry, and saves metric raw depth
(metres), masks, visualizations, and timing/reproducibility metadata. The DAC
notebook also creates the protocol-checked DAC-versus-UniDAC comparison after
both evaluations are available. To reproduce the completed experiment, process
all four G1_A sequences with `FRAME_STEP = 1` and no frame limit: 1,204 frames
per model. The notebooks do not overwrite completed files and produce
per-recording tables plus a weighted four-recording summary.
The dedicated benchmark notebook measures both models sequentially in one GPU
runtime using identical decoded frames, five warm-up runs, and ten timed
repetitions per frame.

## Depth visualization scale

Metric colors use the same fixed **0.5-10.0 m** range for both camera
geometries. This makes the two DAC panels below directly comparable by color;
the colorbar states the unit, fixed policy, and exact limits.

![Shared 0.5-10.0 m scale for fisheye and perspective cameras](docs/images/metric_camera_scale_comparison.png)

Depth Anything V2 is different: it produces relative inverse depth in
arbitrary units and uses a separately labelled per-image 2nd-98th-percentile
range. Its colors cannot be interpreted as metres. See the
[visualization scale policy](docs/VISUALIZATION_SCALE.md) for the full decision
and report-ready wording.

## Example outputs

Each visualization shows the input RGB (left) and predicted depth (right).
Depth is colorized so **near = bright, far = dark**. Neutral gray means the
model produced no valid depth there; it is not a far-away surface. The DAC and
UniDAC examples use the same fixed 0.5-10.0 m scale and can be compared by
color. Depth Anything V2 uses a separately labelled per-image scale in
arbitrary units, so compare only its scene shape and boundaries with the metric
models. Every new plot states its exact scale directly on the colorbar.

### Perspective camera (ZED_B) — model comparison

The same perspective frame with all three models. Depth Anything V2 gives a
sharp visual result but only in relative units. DAC and UniDAC produce metric
depth in metres; their panels share the fixed display scale and are therefore
directly comparable by color.

| Depth Anything V2 (relative) | DAC `dac-indoor-resnet101` (metric) | UniDAC (metric) |
| --- | --- | --- |
| ![Perspective — Depth Anything V2](docs/images/perspective_depth_anything_v2.png) | ![Perspective — DAC indoor ResNet101](docs/images/perspective_dac_indoor_resnet101.png) | ![Perspective — UniDAC](docs/images/perspective_unidac.png) |

### Fisheye camera (G1_A) — model comparison

The same fisheye frame with all three models. Depth Anything V2 (relative
depth) has no explicit fisheye calibration and gives a flat, roughly radial
result with a faint subject. DAC and UniDAC both use the calibrated lens model
and produce metric depth in metres. UniDAC is the accuracy-oriented model
selected by the matched four-recording evaluation.

| Depth Anything V2 (relative) | DAC `dac-indoor-resnet101` (metric) | UniDAC (metric) |
| --- | --- | --- |
| ![Fisheye — Depth Anything V2](docs/images/fisheye_depth_anything_v2.png) | ![Fisheye — DAC indoor ResNet101](docs/images/fisheye_dac_indoor_resnet101.png) | ![Fisheye — UniDAC](docs/images/fisheye_unidac.png) |

**Current baseline:** Depth Anything V2 is the sharp perspective-camera visual
baseline. For calibrated metric G1_A depth, the matched 1,204-frame evaluation
selects UniDAC when 3D localization accuracy is the priority and keeps DAC as
the faster comparison baseline. UniDAC reduces the all-recording mean ZED and
LiDAR 3D errors by 31.5% and 30.5%, while DAC is 2.8 times faster in the
controlled M1 Pro benchmark. See the
[all-recording result](docs/RESULTS_ALL_RECORDINGS_DAC_VS_UNIDAC.md), including
the important UniDAC far-range failure segment in `recording4`.

## Running monocular depth estimation

All commands assume you run from inside `monocular-depth-estimation/`, with the
dataset one level up (so `--data_dir ../` points at the folder holding
`intrinsic.json` and `recording1..4/`).

Each run writes to `outputs/`: an RGB copy (`_rgb.jpg`), a colorized depth
visualization (`_depth.png`), a raw depth array (`_depth_raw.npy`), and — for the
fisheye camera — a binary valid-region mask (`_mask.png`).

### Recommended per camera

```bash
# Perspective camera (ZED_B) — Depth Anything V2 gives the sharpest result
uv run python run_depth.py --data_dir ../ --sensor ZED_B --recording recording1 --image_index 0

# Fisheye camera (G1_A) — accuracy-oriented metric baseline
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 \
  --image_index 0 --model unidac --visualization_range 0.5 10.0

# Fisheye camera (G1_A) — faster metric comparison baseline
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 \
  --image_index 0 --model dac --variant dac-indoor-resnet101 \
  --visualization_range 0.5 10.0
```

> **Note on units.** Depth Anything V2 outputs *relative* (inverse) depth in
> arbitrary units. DAC outputs *metric* depth in metres — use it when you need
> real distances for 3D projection / LiDAR comparison. UniDAC also outputs
> metric depth in metres.

### Hugging Face authentication

Model downloads use `HF_TOKEN` when it is set in the shell or in the project's
git-ignored `.env` file. For example: `HF_TOKEN=hf_your_token`. The token is
used for both Transformers and direct Hugging Face Hub downloads, and is never
printed.

## LiDAR ground-truth evaluation (perspective camera)

`run_depth.py` can timestamp-match a LiDAR cloud, transform its points into the
perspective camera frame, project the visible points using the calibrated OpenCV
intrinsics (including distortion), and compare the model at those pixels.

```bash
uv run python run_depth.py --data_dir ../ --sensor ZED_B --recording recording1 \
  --image_index 0 --evaluate_lidar --lidar_sensor YOUR_LIDAR_SENSOR
```

For a recording-level result, add `--evaluate_lidar_all`. It evaluates every
RGB frame with a LiDAR timestamp match, skips unmatched frames, and writes
`recording1_ZED_B_YOUR_LIDAR_SENSOR_lidar_global_metrics.json`. The global file
combines the per-frame evaluations by LiDAR-point count without refitting an
additional global alignment; per-frame metrics and diagnostic files are also
saved.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python run_depth.py \
  --data_dir .. \
  --output_dir ../local_experiments/perspective/unidac/recording1 \
  --sensor ZED_B \
  --recording recording1 \
  --model unidac \
  --device mps \
  --evaluate_lidar_all \
  --lidar_sensor E1_A \
  --alignment none
```

Replace `YOUR_LIDAR_SENSOR` with the sensor key used both in
`extrinsics.json` and in `recording1/data/` (for example, an Ouster/LiDAR
folder name). The default assumes each extrinsic maps **sensor → G1_A**. If
your calibration stores the inverse direction, add:

```bash
--extrinsics_convention reference_to_sensor
```

The run writes the following additional files alongside the depth output:

- `_lidar_projection.png`: RGB image with the nearest projected LiDAR return
  per pixel, coloured by LiDAR depth. This is the primary calibration sanity
  check—points should land on their corresponding image structures.
- `_lidar_metrics.json`: count, alignment parameters, MAE, RMSE, AbsRel,
  SqRel, log-RMSE, and the standard δ<1.25 / δ<1.25² / δ<1.25³ accuracies.
- `_lidar_samples.csv`: pixel coordinates and each LiDAR/predicted depth pair,
  including signed and absolute error for plotting or further analysis.
- `_signed_error_overlay.png`: with `--plot_evaluation`, projected samples
  coloured by signed `prediction - LiDAR` error. Blue means that the predicted
  surface is closer than the LiDAR return, red means farther, and white means
  agreement. The default fixed range is -2.0 to +2.0 m.

Depth Anything V2 is not metrically scaled. For it, the default `--alignment
auto` takes the reciprocal of its inverse-depth output and estimates a
least-squares scale against the sampled LiDAR depths **before** calculating
errors. This is a **per-frame, LiDAR-aligned** score, not a zero-shot metric-depth
score. Use `--alignment median` for median scaling or `--alignment none` only
for a model that already outputs metres (such as DAC or UniDAC). Record the
selected alignment with every result; fitting and testing on the same sparse
points is useful for evaluating shape/relative-depth quality but is optimistic
for metric-depth claims.

After running all recordings, validate the protocol and calculate a correctly
point-weighted suite result:

```bash
uv run python summarize_lidar_suite.py \
  --results_root ../local_experiments/perspective/unidac \
  --output_dir ../local_experiments/perspective/analysis \
  --recordings recording1 recording2 recording3 recording4 \
  --sensor ZED_B \
  --lidar_sensor E1_A \
  --model UniDAC \
  --expected_alignment none
```

To rank error outliers and test whether person blur is associated with higher
error, run:

```bash
uv run python analyze_lidar_outliers.py \
  --data_dir .. \
  --results_root ../local_experiments/perspective/unidac \
  --output_dir ../local_experiments/perspective/analysis \
  --recordings recording1 recording2 recording3 recording4 \
  --sensor ZED_B \
  --lidar_sensor E1_A \
  --person_blur \
  --device mps
```

The blur analysis uses variance of the Laplacian inside eroded YOLO person
masks. Lower values mean less person detail. Its Spearman correlation tests an
association with error; it does not establish blur as the cause.

### Signed residual and person distance-error plots

Existing pointwise output can be plotted without rerunning a model:

```bash
uv run python plot_evaluation_diagnostics.py residual-overlay \
  --image outputs/unidac/recording2/recording2_ZED_B_0014_rgb.jpg \
  --samples outputs/unidac/recording2/recording2_ZED_B_0014_lidar_samples.csv \
  --error-limit-m 2.0 \
  --output outputs/unidac/recording2/recording2_ZED_B_0014_signed_error_overlay.png
```

![Signed UniDAC depth residuals at projected LiDAR points](docs/images/unidac_perspective_signed_lidar_residual.png)

Plot 3D person localization error against physical-reference distance by
passing one or more result files per model. Repeating a label combines its
recordings:

```bash
uv run python plot_evaluation_diagnostics.py localization-error \
  --run DAC=outputs/evaluation/dac/recording2/person_measurements.csv \
  --run DAC=outputs/evaluation/dac/recording3/person_measurements.csv \
  --run UniDAC=outputs/evaluation/unidac/recording2/person_measurements.csv \
  --run UniDAC=outputs/evaluation/unidac/recording3/person_measurements.csv \
  --reference lidar \
  --bin-width-m 1.0 \
  --output outputs/evaluation/person_localization_error_vs_lidar_distance.png
```

Each translucent point is one valid person/reference match. The solid line is
the median error in each occupied distance bin. The x-axis uses the Euclidean
range of the saved reference XYZ position, falling back to its median range.

## LiDAR projection (fisheye camera)

`project_lidar_fisheye.py` timestamp-matches one or both E1 LiDARs, transforms
their finite XYZ returns into the G1_A reference frame, and projects them with
the calibrated OpenCV fisheye camera matrix and distortion coefficients. It
uses Euclidean camera range and keeps the nearest return at each image pixel.

```bash
uv run python project_lidar_fisheye.py \
  --data_dir .. \
  --output_dir ../local_experiments/fisheye_lidar_projection/recording1 \
  --recording recording1 \
  --image_index 0 \
  --lidar_sensors E1_A E1_B \
  --lidar_max_dt 0.05 \
  --visualization_range 0.5 10.0
```

Add `--all_frames` for a recording batch; `--frame_step` and `--max_frames`
can select a review subset. The command is resumable and refuses to mix
existing outputs created with different sensors, timestamp settings, distance
limits, color scales, or extrinsics conventions unless `--overwrite` is used.

Every frame produces:

- `_projection.png`: RGB and fixed-scale LiDAR overlay with a range colorbar;
- `_points.csv`: visible sensor, subpixel coordinates, and Euclidean range;
- `_metadata.json`: exact timestamps, offsets, calibration matrices,
  transforms, counts, scale, convention, and project revision.

The recording summary reports matches, timestamp statistics, and projected
point counts per LiDAR. When both LiDARs are shown together, viewpoint-dependent
occlusion shadows can differ; run `--lidar_sensors E1_A` and `E1_B` separately
when validating each calibration.

### Depth Anything V2 (default model)

```bash
# Perspective camera, first frame
uv run python run_depth.py --data_dir ../ --sensor ZED_B --recording recording1 --image_index 0

# Fisheye camera (auto circular valid-region masking)
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 --image_index 0

# Larger encoder (sharper, slower)
uv run python run_depth.py --data_dir ../ --sensor ZED_B --encoder large
```

### Depth Any Camera (DAC) — metric depth, native fisheye support

DAC ([Guo et al., CVPR 2025](https://github.com/yuliangguo/depth_any_camera))
gives zero-shot **metric** depth and handles fisheye intrinsics natively.
If you did not already complete the local setup above, clone and pin the DAC
source. Its weights download from Hugging Face on first run (~700 MB) and are
cached afterwards:

```bash
git clone https://github.com/yuliangguo/depth_any_camera.git third_party/depth_any_camera
git -C third_party/depth_any_camera checkout 371ee299429257bb9a27d1e23b7dc53670e37023
```

The compiled deformable-attention C++ op is **not** required (the ResNet101
configs use `attn_dec=false`); it is mocked automatically.

```bash
# Fisheye, indoor ResNet101  (recommended for this dataset)
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 --image_index 0 --model dac --variant dac-indoor-resnet101

# Perspective, indoor ResNet101
uv run python run_depth.py --data_dir ../ --sensor ZED_B --recording recording1 --image_index 0 --model dac --variant dac-indoor-resnet101

# Indoor SwinL backbone (heavier; can show artifacts at the fisheye periphery)
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 --image_index 0 --model dac --variant dac-indoor-swinl

# Outdoor ResNet101 (KITTI-trained; overestimates indoor distances)
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 --image_index 0 --model dac-outdoor-resnet101
```

Available variants: `dac-indoor-resnet101`, `dac-indoor-swinl`,
`dac-outdoor-resnet101`, `dac-outdoor-swinl`.

### UniDAC — metric depth, Colab GPU recommended

The Colab notebook above installs the pinned official source and checkpoint. A
local run requires the pinned UniDAC checkout from the setup section; if it is
not present, run:

```bash
git clone https://github.com/girish1511/UniDAC.git third_party/UniDAC
git -C third_party/UniDAC checkout 9ddfc1f4cea68e08273ec9bca037f2ef9e1aa90e
```

For the circular G1_A lens, the adapter expands UniDAC's rectangular ERP crop
to cover the full vertical fisheye field of view and masks the black lens border
before inference. The saved metadata records the effective ERP crop and FoV.
For later sensor/model evaluation, `create_common_valid_depth_mask()` builds the
intersection mask so every error metric uses exactly the same aligned pixels.
After the same setup is available locally, the single-frame CLI also accepts:

```bash
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 --image_index 0 --model unidac
```

## 3D person tracking and physical-sensor evaluation

After generating a G1_A metric-depth batch, run the automated evaluator:

```bash
uv run python evaluate_person_tracking.py \
  --data_dir .. \
  --depth_output_dir /path/to/cv_project_outputs/unidac \
  --output_dir outputs/evaluation/recording1 \
  --recording recording1 \
  --frame_step 10 \
  --max_frames 50
```

The evaluator uses the COCO-pretrained
[YOLO26 instance-segmentation model](https://docs.ultralytics.com/tasks/segment/)
for person masks and [ByteTrack](https://docs.ultralytics.com/modes/track/) for
persistent IDs. For every tracked person it:

1. takes robust metric model depth inside an eroded instance mask;
2. converts the mask centroid and Euclidean ray depth to G1_A camera-frame XYZ;
3. reprojects the closest ZED_B depth image through the calibrated extrinsics;
4. reprojects the two closest LiDAR sweeps and rejects returns inconsistent
   with the stereo-visible foreground, avoiding cross-sensor occlusion errors;
5. computes common-pixel MAE, RMSE, absolute-relative error, bias, and 3D error.

Outputs are written to the evaluation directory:

- `person_measurements.csv`: one row per detected person and frame;
- `track_summary.csv`: aggregate measurements per persistent track ID;
- `evaluation_summary.json`: run configuration, versions, counts, and averages;
- `annotated/`: masks, boxes, IDs, and model/ZED/LiDAR distances.

The updated Colab notebook exposes the same workflow in its optional final
section and caches `yolo26n-seg.pt` in Google Drive.

To compare evaluation summaries from UniDAC and DAC using the same protocol:

```bash
uv run python compare_evaluation_runs.py \
  --run UniDAC=/path/to/unidac/evaluation_summary.json \
  --run DAC=/path/to/dac/evaluation_summary.json \
  --output_markdown outputs/evaluation/model_comparison.md
```

See the [experiment guide](docs/EXPERIMENT_GUIDE.md) for the complete sampling,
timing, visualization, and reporting protocol.

## All depth-estimation options

```bash
uv run python run_depth.py \
  --data_dir ../ \
  --sensor G1_A \
  --recording recording1 \
  --image_index 0 \
  --model dac \
  --variant dac-indoor-resnet101 \
  --fisheye_mask auto \
  --invalid_value nan \
  --visualization_range 0.5 10.0
```

- Omit `--model` to use the default `depth_anything_v2`.
- `--sensor` accepts `ZED_B` or `G1_A`.
- `--model` accepts `depth_anything_v2`, `dac`, `unidac`, or a full DAC variant.
- `--variant` applies only when `--model dac`; `--encoder` accepts `small`,
  `base`, or `large` and applies only to Depth Anything V2.
- Run `uv run python run_depth.py --help` for LiDAR-evaluation, device, masking,
  alignment, and output options.
