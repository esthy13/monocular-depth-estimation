# Experiment guide

This guide is the reproducible path from camera images to metric person tracks
and physical-sensor error measurements. Run commands from the repository root.

## 1. Environment and data

Install the locked local environment:

```bash
uv sync --locked
```

The data directory must contain `intrinsic.json`, `extrinsics.json`, and the
`recording1` to `recording4` folders. The pipeline treats G1_A as the reference
camera coordinate system.

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

Use `FRAME_STEP = 1` and `MAX_FRAMES_PER_SENSOR = None` only for the final full
sequence. The completed matched runs used all 133 frames from `recording1`. The
next full experiment covers the remaining recordings with:

```python
RUN_BATCH = True
BATCH_RECORDINGS = ["recording2", "recording3", "recording4"]
BATCH_SENSORS = ["G1_A"]
FRAME_STEP = 1
MAX_FRAMES_PER_SENSOR = None
OVERWRITE = False
```

This selects 124, 791, and 156 G1_A frames respectively: 1,071 frames per
model. Run UniDAC first, then DAC. The notebooks resume from existing metadata
and raw-depth files if Colab disconnects.

The notebooks store raw metric depth and metadata under their separate roots:

```text
MyDrive/cv_project_outputs/unidac/<recording>/G1_A/
MyDrive/cv_project_outputs/dac/<recording>/G1_A/
```

## 4. Use comparable plot scales

The Colab notebook fixes every metric plot to `0.5–10.0 m`. The CLI exposes the
same behavior:

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
colorbar is labelled `Depth (m)`. Depth Anything V2 is relative depth and is
labelled `Depth (a.u.)`; its colors must not be numerically compared with DAC or
UniDAC.

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

After `recording2`–`recording4` are complete for both models, the DAC notebook's
comparison cell also calls `compare_evaluation_suite.py`. It combines
`recording1`–`recording4`, weighting accuracy metrics by the number of valid
physical-reference detections and timing by processed frames. The resulting
CSV and Markdown report are stored under:

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
