# Experiment guide

This standalone guide is the reproducible path from camera images to metric
person tracks and physical-sensor error measurements. Run commands from the
repository root. The commands and paths were checked against the
current implementation after the completed four-recording experiment.

## 1. Environment and data

Install the locked local environment:

```bash
uv sync --locked
```

The data directory must contain `intrinsic.json`, `extrinsics.json`, and the
`recording1` to `recording4` folders. The pipeline treats G1_A as the reference
camera coordinate system. With the repository inside `cv_project_data/`, the
expected layout is:

```text
cv_project_data/
├── intrinsic.json
├── extrinsics.json
├── recording1/ ... recording4/
└── monocular-depth-estimation/
```

For local DAC and UniDAC execution, install the pinned upstream sources used by
the reference experiment. They are ignored by Git:

```bash
git clone https://github.com/yuliangguo/depth_any_camera.git third_party/depth_any_camera
git -C third_party/depth_any_camera checkout 371ee299429257bb9a27d1e23b7dc53670e37023

git clone https://github.com/girish1511/UniDAC.git third_party/UniDAC
git -C third_party/UniDAC checkout 9ddfc1f4cea68e08273ec9bca037f2ef9e1aa90e
```

The official checkpoints download from Hugging Face on first use. Set
`HF_TOKEN` in the shell or the git-ignored `.env` file only if authentication is
required; never commit a token.

Run UniDAC with
[`notebooks/unidac_colab.ipynb`](../notebooks/unidac_colab.ipynb) and the matched
DAC baseline with [`notebooks/dac_colab.ipynb`](../notebooks/dac_colab.ipynb).
Both notebooks contain the private-repository token instructions, pinned model
source and checkpoint, a one-frame smoke test, a resumable batch, timing, and
the optional person evaluation.

Use [`notebooks/benchmark_colab.ipynb`](../notebooks/benchmark_colab.ipynb) for
the controlled final latency comparison. It benchmarks both models in the same
Colab process and on the same GPU.

## 2. Smoke test before a batch

In Colab, select a GPU runtime and run the notebook from top to bottom. The
dependency cell restarts the runtime once; run all cells again after it
reconnects. Keep the initial settings at one frame until the raw depth, gray
invalid region, metadata, and visualization look correct.

The smoke-test and batch timings are run diagnostics, not the final speed
measurement. The dedicated benchmark notebook discards five warm-up runs before
recording repeated timings.

## 3. Generate matched G1_A batches

Start with the same sampled frames for every model:

```python
RUN_BATCH = True
BATCH_RECORDINGS = ["recording1"]
BATCH_SENSORS = ["G1_A"]
FRAME_STEP = 10
MAX_FRAMES_PER_SENSOR = 50
OVERWRITE = False
```

Use `FRAME_STEP = 1` and `MAX_FRAMES_PER_SENSOR = None` for a full sequence. The
completed matched experiment used all 133 frames from `recording1` and the
following settings for the other three recordings:

```python
RUN_BATCH = True
BATCH_RECORDINGS = ["recording2", "recording3", "recording4"]
BATCH_SENSORS = ["G1_A"]
FRAME_STEP = 1
MAX_FRAMES_PER_SENSOR = None
OVERWRITE = False
```

This selects 124, 791, and 156 G1_A frames respectively: 1,071 additional
frames and 1,204 total frames per model. To reproduce the comparison, use the
same settings for UniDAC and DAC. The notebooks resume from existing metadata
and raw-depth files if Colab disconnects.

The notebooks store raw metric depth and metadata under their separate roots:

```text
MyDrive/cv_project_outputs/unidac/<recording>/G1_A/
MyDrive/cv_project_outputs/dac/<recording>/G1_A/
```

## 4. Use comparable plot scales

The Colab notebooks and CLI fix every metric plot to `0.5-10.0 m` for both the
G1_A fisheye and ZED_B perspective cameras. The CLI uses this metric default
even if the option is omitted; pass it explicitly in recorded experiment
commands for clarity:

```bash
uv run python run_depth.py \
  --data_dir .. \
  --sensor G1_A \
  --recording recording1 \
  --image_index 0 \
  --model dac \
  --variant dac-indoor-resnet101 \
  --visualization_range 0.5 10.0
```

Use the same range for every metric camera/model plot in one comparison. Each
colorbar states `Metric depth (m)`, `Fixed display range`, and its exact limits.
Depth Anything V2 is relative inverse depth: its colorbar states `a.u.`, the
per-image percentile policy, and the exact limits. Its colors must not be
numerically compared with DAC or UniDAC. The rationale, comparison figure, and
report-ready wording are in the
[depth visualization scale policy](VISUALIZATION_SCALE.md).

## 5. Run person tracking and sensor evaluation

The optional evaluation cell loops over every entry in
`EVALUATION_RECORDINGS`. After the matching G1_A batch finishes, set:

```python
RUN_PERSON_EVALUATION = True
EVALUATION_RECORDINGS = ["recording2", "recording3", "recording4"]
```

The equivalent command is:

```bash
uv run python evaluate_person_tracking.py \
  --data_dir .. \
  --depth_output_dir /path/to/cv_project_outputs/<model> \
  --output_dir outputs/evaluation/<model>-recording1 \
  --recording recording1 \
  --frame_step 10 \
  --max_frames 50
```

This produces:

- `person_measurements.csv`: every tracked person and frame;
- `track_summary.csv`: per-track aggregate values;
- `evaluation_summary.json`: configuration, model versions, accuracy, and speed;
- `annotated/`: review images containing IDs and reference distances.

The evaluator can process any metric model stored in the same output layout. It
reads the model name and saved depth inference time from each metadata JSON.

### Plot the two physical-reference diagnostics

`run_depth.py --plot_evaluation` now saves a signed residual overlay for every
evaluated perspective frame. The residual is defined as prediction minus LiDAR
depth. Blue points are predicted closer than LiDAR, red points are predicted
farther, and white points agree. The default fixed range is -2.0 to +2.0 m;
change it with `--residual_error_limit_m` only when the plot states the new
range.

An overlay can also be recreated from an existing RGB image and its pointwise
CSV without rerunning inference:

```bash
uv run python plot_evaluation_diagnostics.py residual-overlay \
  --image outputs/perspective/unidac/recording2/recording2_ZED_B_0014_rgb.jpg \
  --samples outputs/perspective/unidac/recording2/recording2_ZED_B_0014_lidar_samples.csv \
  --error-limit-m 2.0 \
  --output outputs/plots/unidac_recording2_frame14_signed_error.png
```

Create the person plot from the matched per-detection measurements:

```bash
uv run python plot_evaluation_diagnostics.py localization-error \
  --run DAC=outputs/evaluation/dac/recording2/person_measurements.csv \
  --run DAC=outputs/evaluation/dac/recording3/person_measurements.csv \
  --run DAC=outputs/evaluation/dac/recording4/person_measurements.csv \
  --run UniDAC=outputs/evaluation/unidac/recording2/person_measurements.csv \
  --run UniDAC=outputs/evaluation/unidac/recording3/person_measurements.csv \
  --run UniDAC=outputs/evaluation/unidac/recording4/person_measurements.csv \
  --reference lidar \
  --bin-width-m 1.0 \
  --output outputs/plots/person_localization_error_vs_lidar_distance.png
```

The x-axis is ground-truth person range from the saved LiDAR XYZ position and
the y-axis is Euclidean 3D localization error. Translucent points show every
valid match; lines show the median error in each occupied 1 m distance bin.

## 6. Compare UniDAC and DAC

Evaluate both models on exactly the same recording, start index, frame step,
and maximum frame count. Then create the report table:

```bash
uv run python compare_evaluation_runs.py \
  --run UniDAC=outputs/evaluation/unidac-recording1/evaluation_summary.json \
  --run DAC=outputs/evaluation/dac-recording1/evaluation_summary.json \
  --output_csv outputs/evaluation/model_comparison.csv \
  --output_markdown outputs/evaluation/model_comparison.md
```

The command warns when frame counts or sampling settings differ. Do not claim
one model is better unless the compared summaries use the same protocol.

For the completed four-recording result, combine the matched evaluation roots:

```bash
uv run python compare_evaluation_suite.py \
  --model_root UniDAC=outputs/evaluation/unidac \
  --model_root DAC=outputs/evaluation/dac \
  --recordings recording1 recording2 recording3 recording4 \
  --output_csv outputs/evaluation/all_recordings/model_comparison.csv \
  --output_markdown outputs/evaluation/all_recordings/model_comparison.md
```

The tool expects each model root to contain
`<recording>/evaluation_summary.json`. It weights accuracy metrics by the number
of valid physical-reference detections and timing by processed frames. The DAC
notebook runs the same command and stores its CSV and Markdown report under:

```text
MyDrive/cv_project_outputs/comparison/all_recordings/
```

## 7. Run the controlled speed benchmark

Open `notebooks/benchmark_colab.ipynb` in a fresh GPU runtime and run all cells.
Its fixed protocol is:

```python
RECORDING = "recording1"
SENSOR = "G1_A"
FRAME_INDICES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 132]
WARMUP_RUNS = 5
TIMED_RUNS_PER_FRAME = 10
MODEL_ORDER = ["DAC", "UniDAC"]
```

The notebook decodes the images once, loads each model sequentially on the same
GPU, synchronizes CUDA around every measurement, and times only preprocessing,
neural inference, and camera back-projection. It writes:

- `benchmark_timings.csv`: all 200 individual measurements;
- `benchmark_summary.json`: protocol, provenance, distribution statistics, and
  the median speed ratio;
- `benchmark_report.md`: report-ready table and conclusion.

Model loading, image loading, person detection, tracking, and disk output are
outside the timed region.

## 8. Run the perspective metric-depth LiDAR evaluation

Use explicit `--alignment none` for a metric model. On Apple Silicon, process
one recording at a time so each finished recording has its own resumable result
directory:

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

Repeat for `recording2` to `recording4`, changing both recording arguments.
Then run `summarize_lidar_suite.py` to verify that every global JSON is a true
metric evaluation and combine its point-level statistics. Run
`analyze_lidar_outliers.py --person_blur` to produce the per-frame ranking and
the person-mask sharpness/error correlations. The exact commands and output
descriptions are in the README LiDAR section.

Do not compare this zero-shot metric result directly as if it used the same
protocol as a Depth Anything V2 run aligned to the evaluation LiDAR points.
That fitted run is a relative-depth/oracle diagnostic, whereas `alignment=none`
measures metric scale without using the test targets to fit each frame.

For the G1_A projection task, generate a direct calibration audit image before
using LiDAR points as person references:

```bash
uv run python project_lidar_fisheye.py \
  --data_dir .. \
  --output_dir ../local_experiments/fisheye_lidar_projection/recording1 \
  --recording recording1 \
  --image_index 0 \
  --lidar_sensors E1_A E1_B \
  --visualization_range 0.5 10.0
```

Inspect E1_A and E1_B individually as well as together. Static boundaries such
as walls, doors, and floor edges should align. A moving person and the empty
region behind them can differ between sensors because the LiDARs, camera, and
rotating scans do not share one optical centre or exposure instant. The person
evaluator therefore uses the projected stereo reference to reject inconsistent
background returns.

## 9. Final experiment protocol

For the report, record all of the following:

- project commit and upstream model commit;
- model checkpoint and variant;
- recording, selected frames, and camera;
- fixed visualization range;
- GPU name, model order, warm-up count, timed-run count, and timing scope;
- person detections and persistent tracks;
- common-pixel ZED MAE, RMSE, AbsRel, bias, and 3D error;
- stereo-gated LiDAR point count and 3D error;
- failures, missing references, and synchronization offsets.

The matched four-recording comparison and controlled Apple Metal benchmark are
complete. See
[`RESULTS_ALL_RECORDINGS_DAC_VS_UNIDAC.md`](RESULTS_ALL_RECORDINGS_DAC_VS_UNIDAC.md)
for the aggregate result, per-recording limitation, and final model
recommendation. A Tesla T4 benchmark should still be reported separately if a
Colab-specific speed claim is needed.

## 10. Local VS Code alternative

On Apple Silicon, use `run_metric_depth_batch.py --device mps` from the VS Code
terminal. The command is resumable and stores only the raw metric arrays and
metadata needed by `evaluate_person_tracking.py`; annotated evaluation images
are created later by the evaluator. The Colab and local outputs use the same
directory and filename contract, so the comparison tools work with either.

The local Metal benchmark uses the same `benchmark_depth_models.py` protocol as
Colab with `--device mps`. Do not mix its latency measurements with the earlier
Tesla T4 timings; report hardware-specific speed tables separately.

## 11. Documentation and code checks

Before sharing a new result or changing the board tasks to Done, run:

```bash
uv run python run_depth.py --help
uv run python evaluate_person_tracking.py --help
uv run python compare_evaluation_suite.py --help
PYTHONPATH=. uv run --with pytest pytest -q tests
```

The `PYTHONPATH=.` prefix is needed when invoking the temporary `pytest` console
script so imports resolve from the repository root. The test-only dependency is
provided transiently by `uv run --with` and is not added to the runtime lockfile.
