# monocular-depth-estimation

Metric monocular depth, calibrated 3D person localization, and tracking for the
G1_A fisheye camera, evaluated against ZED stereo and two LiDAR sensors.

Detailed instructions and methodology:

- [Experiment and execution guide](docs/EXPERIMENT_GUIDE.md)
- [Camera geometry used by each model](docs/MODEL_GEOMETRY.md)
- [Recording 1 UniDAC full-sequence results](docs/RESULTS_RECORDING1_UNIDAC.md)

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
├── evaluate_person_tracking.py # 3D person/sensor evaluation
└── compare_evaluation_runs.py  # UniDAC/DAC report table
```

## Local setup

Install Python 3.12 and `uv`, then create the locked environment:

```bash
uv sync --locked
```

Run commands with `uv run python ...`; manual activation is optional.

## Google Colab: UniDAC on GPU

Use the ready-to-run notebook for the newer
[UniDAC](https://github.com/girish1511/UniDAC) metric-depth model:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/esthy13/monocular-depth-estimation/blob/depth-pipeline/notebooks/unidac_colab.ipynb)

Because this repository is private, an unauthenticated Colab link returns 404.
Open the notebook from Colab's GitHub picker after authorizing private-repository
access, or download `notebooks/unidac_colab.ipynb` while signed in to GitHub and
choose **File → Upload notebook** in Colab.

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

The notebook pins the official UniDAC source, caches the ~1.4 GB checkpoint in
Drive, uses the calibrated G1_A/ZED_B camera geometry, and saves metric raw depth
(metres), masks, visualizations, and timing/reproducibility metadata.

## Example outputs

Each visualization shows the input RGB (left) and predicted depth (right).
Depth is colorized so **near = bright, far = dark**. Neutral gray means the
model produced no valid depth there; it is not a far-away surface.

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

**Current baseline:** Depth Anything V2 is the sharp perspective-camera
visual baseline, while DAC is the established calibrated metric baseline for
G1_A. UniDAC is the newer metric candidate used by the automated evaluation.
The final recommendation must come from the same-frame DAC/UniDAC sensor
comparison rather than these qualitative images.

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

# Fisheye camera (G1_A) — calibrated metric DAC baseline with a fixed report scale
uv run python run_depth.py --data_dir ../ --sensor G1_A --recording recording1 --image_index 0 --model dac --variant dac-indoor-resnet101 --visualization_range 0.5 10.0
```

> **Note on units.** Depth Anything V2 outputs *relative* (inverse) depth in
> arbitrary units. DAC outputs *metric* depth in metres — use it when you need
> real distances for 3D projection / LiDAR comparison. UniDAC also outputs
> metric depth in metres.

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

### UniDAC — metric depth, Colab GPU recommended

The Colab notebook above installs the pinned official source and checkpoint.
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
  --sensor {ZED_B|G1_A} \           # perspective | fisheye
  --recording recording1 \         # recording1..4
  --image_index 0 \                # frame index
  --model {depth_anything_v2|dac|unidac} \
  --variant dac-indoor-resnet101 \ # only with --model dac
  --encoder {small|base|large} \   # only for depth_anything_v2
  --fisheye_mask {auto|none} \     # auto-masks the lens circle on fisheye
  --invalid_value {nan|zero} \     # value written to masked pixels
  --visualization_range 0.5 10.0   # shared report scale; optional
```

- Omit `--model` to use the default `depth_anything_v2`.
- `--variant` applies only when `--model dac`; `--encoder` only for Depth Anything V2.
