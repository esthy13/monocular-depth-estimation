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

UniDAC should be run with the GPU notebook
[`notebooks/unidac_colab.ipynb`](../notebooks/unidac_colab.ipynb). The notebook
contains the private-repository token instructions, pinned model source and
checkpoint, a one-frame smoke test, a resumable batch, timing, and the optional
person evaluation.

## 2. Smoke test before a batch

In Colab, select a GPU runtime and run the notebook from top to bottom. The
dependency cell restarts the runtime once; run all cells again after it
reconnects. Keep the initial settings at one frame until the raw depth, gray
invalid region, metadata, and visualization look correct.

For a reliable speed measurement, change:

```python
WARMUP_RUNS = 2
TIMED_RUNS = 10
```

The first forward pass includes one-time initialization and must not be used as
the reported inference speed.

## 3. Generate the sampled G1_A batch

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
sequence. The notebook stores raw metric depth and metadata under:

```text
MyDrive/cv_project_outputs/unidac/<recording>/G1_A/
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

The notebook's final optional cell runs the evaluator on the generated G1_A
batch. Set:

```python
RUN_PERSON_EVALUATION = True
```

The equivalent command is:

```bash
uv run python evaluate_person_tracking.py \
  --data_dir .. \
  --depth_output_dir /path/to/cv_project_outputs/unidac \
  --output_dir outputs/evaluation/unidac-recording1 \
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

## 7. Final experiment protocol

For the report, record all of the following:

- project commit and upstream model commit;
- model checkpoint and variant;
- recording, selected frames, and camera;
- fixed visualization range;
- GPU name, warm-up count, and timed-run count;
- person detections and persistent tracks;
- common-pixel ZED MAE, RMSE, AbsRel, bias, and 3D error;
- stereo-gated LiDAR point count and 3D error;
- failures, missing references, and synchronization offsets.

The current one-frame result is a smoke test, not the final quantitative
conclusion. Final claims require the sampled comparison followed by the full
multi-frame run.
