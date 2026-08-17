# monocular-depth-estimation

## Repository structure

```md
monocular-depth-estimation/
├── minutes/
└── references/
    └── res.bib
```

# To start the project
Check that you have python 3.12 and uv installed on your machine already, then create
and start a virtual environment: 
```bash
uv venv
source .venv/bin/activate
```
How to add dependencies to the project:
```
uv add torch
```

## Example outputs

Each visualization shows the input RGB (left) and predicted depth (right).
Depth is colorized so **near = bright, far = dark**.

### Perspective camera (ZED_B) — model difference

The same perspective frame with two models. Depth Anything V2 gives the sharpest,
cleanest result (crisp subject and room edges) but in relative units. DAC is
**metric** (metres) and works across camera types, but is softer on standard
pinhole images.

| Depth Anything V2 (relative) | DAC `dac-indoor-resnet101` (metric) |
| --- | --- |
| ![Perspective — Depth Anything V2](docs/images/perspective_depth_anything_v2.png) | ![Perspective — DAC indoor ResNet101](docs/images/perspective_dac_indoor_resnet101.png) |

### Fisheye camera (G1_A) — model difference

The same fisheye frame with two models. Depth Anything V2 (relative depth) does
not understand fisheye distortion and gives a flat, roughly radial result with a
faint subject. Depth Any Camera (DAC) is distortion-aware and **metric** (metres),
resolving the standing person and the hallway structure.

| Depth Anything V2 (relative) | DAC `dac-indoor-resnet101` (metric) |
| --- | --- |
| ![Fisheye — Depth Anything V2](docs/images/fisheye_depth_anything_v2.png) | ![Fisheye — DAC indoor ResNet101](docs/images/fisheye_dac_indoor_resnet101.png) |

**Takeaway:** use Depth Anything V2 for the perspective camera (sharpest), and
DAC `dac-indoor-resnet101` for the fisheye camera (handles distortion + metric scale).

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

# Fisheye camera (G1_A) — DAC indoor ResNet101 is the best for this indoor data
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 --image_index 0 --model dac --variant dac-indoor-resnet101
```

> **Note on units.** Depth Anything V2 outputs *relative* (inverse) depth in
> arbitrary units. DAC outputs *metric* depth in metres — use it when you need
> real distances for 3D projection / LiDAR comparison.

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

Replace `YOUR_LIDAR_SENSOR` with the sensor key used both in
`extrinsics.json` and in `recording1/data/` (for example, an Ouster/LiDAR
folder name). The default assumes each extrinsic maps **sensor → G1_A**. If
your calibration stores the inverse direction, add:

```bash
--extrinsics_convention reference_to_sensor
```

The run writes three additional files alongside the depth output:

- `_lidar_projection.png`: RGB image with the nearest projected LiDAR return
  per pixel, coloured by LiDAR depth. This is the primary calibration sanity
  check—points should land on their corresponding image structures.
- `_lidar_metrics.json`: count, alignment parameters, MAE, RMSE, AbsRel,
  SqRel, log-RMSE, and the standard δ<1.25 / δ<1.25² / δ<1.25³ accuracies.
- `_lidar_samples.csv`: pixel coordinates and each LiDAR/predicted depth pair,
  for plotting or further analysis.

Depth Anything V2 is not metrically scaled. For it, the default `--alignment
auto` takes the reciprocal of its inverse-depth output and estimates a
least-squares scale against the sampled LiDAR depths **before** calculating
errors. This is a **per-frame, LiDAR-aligned** score, not a zero-shot metric-depth
score. Use `--alignment median` for median scaling or `--alignment none` only
for a model that already outputs metres (such as DAC). Record the selected
alignment with every result; fitting and testing on the same sparse points is
useful for evaluating shape/relative-depth quality but is optimistic for
metric-depth claims.

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
One-time setup (clone the repo; weights download from HuggingFace on first run,
~700 MB, cached afterwards):

```bash
git clone https://github.com/yuliangguo/depth_any_camera.git third_party/depth_any_camera
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

### All options

```bash
uv run python run_depth.py \
  --data_dir ../ \
  --sensor {ZED_B|G1_A} \           # perspective | fisheye
  --recording recording1 \         # recording1..4
  --image_index 0 \                # frame index
  --model {depth_anything_v2|dac} \
  --variant dac-indoor-resnet101 \ # only with --model dac
  --encoder {small|base|large} \   # only for depth_anything_v2
  --fisheye_mask {auto|none} \     # auto-masks the lens circle on fisheye
  --invalid_value {nan|zero}       # value written to masked pixels
```

- Omit `--model` to use the default `depth_anything_v2`.
- `--variant` applies only when `--model dac`; `--encoder` only for Depth Anything V2.
