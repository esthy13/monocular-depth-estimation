# Depth Anything V2–LiDAR evaluation: draft report

Date: 2026-08-18

## Scope and provenance correction

The evaluated run used **Depth Anything V2 (DAV2)**, not UniDepth. In
`run_depth.py`, the default value of `--model` is `depth_anything_v2`; the
repository's model factory contains DAV2 and Depth Any Camera implementations,
but no UniDepth implementation. The command used for these results did not
override the default model. The earlier heading in `minutes/17_08_results.md`
has therefore been corrected.

The saved global JSON files do not currently record the model name or checkpoint,
so the model identity is established from the command and code path rather than
from self-contained output provenance. Future evaluation outputs should record
both fields.

## Alignment and interpretation

All four inspected files
`outputs/recording{1..4}_ZED_B_E1_A_lidar_global_metrics.json` report:

- `alignment_method: inverse_least_squares`
- `alignment_scope: per-frame alignment; point-weighted aggregation without refitting`
- `evaluation_role: oracle_diagnostic_not_metric_accuracy`

This confirms that `--alignment auto` selected `inverse_least_squares` for DAV2.
For each frame, the evaluator fits scale and shift to that frame's LiDAR targets
and converts the inverse-depth-like DAV2 output using

```text
depth = 1 / (scale * prediction + shift)
```

The same LiDAR samples are then scored. This is evaluation leakage by design and
is useful only as an **oracle-aligned depth-shape diagnostic**. In particular,
recording 2's AbsRel of 0.066 is not evidence of 6.6% standalone metric-depth
accuracy. It describes relative-depth shape after ground-truth-assisted,
per-frame alignment. The nested global `metrics.alignment: none` means only that
the already aligned per-frame samples were pooled without a second fit; it does
not mean that the results are unaligned.

## Results

All figures below are per-frame oracle-aligned DAV2 diagnostics.

| Recording | Valid frames | Valid points | MAE (m) | RMSE (m) | RMSE/MAE | AbsRel | delta1 | delta2 | delta3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recording1 | 66 | 530,505 | 0.614 | 3.183 | 5.18 | 0.158 | 0.827 | 0.922 | 0.951 |
| recording2 | 65 | 530,382 | 0.305 | 0.850 | 2.79 | 0.066 | 0.948 | 0.978 | 0.984 |
| recording3 | 1,182 | 9,550,732 | 0.340 | 2.403 | 7.06 | 0.085 | 0.927 | 0.973 | 0.981 |
| recording4 | 201 | 1,651,617 | 0.322 | 0.831 | 2.58 | 0.075 | 0.935 | 0.971 | 0.980 |

Here, delta1, delta2, and delta3 are the fractions satisfying the standard
depth-ratio thresholds 1.25, 1.25 squared, and 1.25 cubed, respectively.

## Synchronization check

The saved results used a 0.050 s matching threshold and `time_offset_s: 0.0`.

| Recording | Mean dt (s) | Median dt (s) | Std dt (s) | Min dt (s) | Max dt (s) |
|---|---:|---:|---:|---:|---:|
| recording1 | +0.00524 | +0.00517 | 0.02527 | -0.02850 | +0.03896 |
| recording2 | +0.00735 | +0.00693 | 0.02631 | -0.02685 | +0.04056 |
| recording3 | +0.01040 | +0.01072 | 0.02721 | -0.03227 | +0.04599 |
| recording4 | -0.01007 | -0.00969 | 0.02686 | -0.04356 | +0.03306 |

The means are close to zero and the standard deviations are consistently about
0.026 s. There is no evidence here of a large fixed timestamp bias. However, no
offset was actually applied, so these statistics do **not** demonstrate that an
offset-calibration procedure succeeded; they demonstrate the residual deltas
from nearest-neighbour matching at zero configured offset. Motion-related error
within the approximately 26 ms spread is still possible.

## Why RMSE is much larger than MAE

The point-level error distributions are strongly heavy-tailed. The evidence
supports the conclusion that a small minority of extreme errors, rather than a
uniform degradation of all predictions, dominates squared error:

| Recording | Median abs. error (m) | p99 (m) | Maximum (m) | Squared error from worst 1% | Points with error >5 m |
|---|---:|---:|---:|---:|---:|
| recording1 | 0.170 | 6.326 | 693.063 | 92.0% | 1.57% |
| recording2 | 0.150 | 5.041 | 26.222 | 66.9% | 1.04% |
| recording3 | 0.164 | 4.776 | 4,124.558 | 94.7% | 0.91% |
| recording4 | 0.168 | 4.163 | 26.366 | 60.4% | 0.73% |

The worst 0.1% alone contributes 56.5% of squared error in recording1 and 89.4%
in recording3. This explains their unusually high RMSE/MAE ratios of 5.18 and
7.06.

### Confirmed numerical failure mode

The most extreme errors in recordings 1 and 3 are caused by the per-frame
inverse-depth conversion approaching a pole. A fitted negative shift makes
`scale * prediction + shift` nearly zero for a few samples, so its reciprocal
becomes extremely large:

- recording1 frame 56: scale 0.280050, shift -0.089117, maximum aligned depth
  702.003 m for LiDAR depth 8.940 m; frame RMSE 25.210 m.
- recording3 frame 394: scale 0.205745, shift -0.019848, maximum aligned depth
  4,136.975 m for LiDAR depth 12.417 m; frame RMSE 65.737 m.
- Related poles occur in recording3 frames 392, 663, and 664, with aligned
  predictions between about 939 m and 1,832 m.

These samples are not ordinary far-range metric errors. They are instability in
the oracle inverse alignment and should be reported, not silently clipped or
removed. A robust diagnostic should record the minimum fitted denominator and
flag fits whose valid prediction range crosses or approaches zero.

### Depth-range dependence

Farther points are also disproportionately influential. For recording3, points
with LiDAR depth in (10, 20] m are only 0.41% of samples but contribute 76.1% of
total squared error; this bin includes the inverse-alignment poles above. In
recording1, the (6, 10] m bin contains 27.8% of samples and contributes 92.4% of
squared error. In recordings 2 and 4, the small set of LiDAR points above 20 m
has errors near 22–26 m and contributes 20.8% and 15.6% of squared error,
respectively.

This establishes a correlation with depth range, but it does not by itself
identify the physical cause. Occlusion, camera–LiDAR parallax, and sampling across
depth discontinuities are plausible explanations for some far-range mismatches,
especially because the sensors have different viewpoints. They are **not
confirmed** by the saved point CSVs, which contain no occlusion labels or image
boundary classification. Confirmation requires inspection of synchronized RGB,
projected LiDAR, and residual overlays, plus boundary/non-boundary statistics.

## Defensible conclusions and next steps

The projection pipeline appears geometrically consistent based on the existing
validation work, and synchronization shows no large fixed bias. The current
numbers nevertheless cannot be presented as metric-depth accuracy because DAV2
is non-metric and alignment is fitted per frame on the evaluation targets.

The strongest supported conclusion is: **DAV2 preserves useful relative-depth
shape after oracle alignment, but its metric generalization has not been measured
by this run. RMSE is dominated by sparse tails, including confirmed reciprocal
singularities in the inverse-alignment diagnostic and a strong association with
farther LiDAR ranges.**

Recommended follow-up work:

1. Report three modes separately: unaligned DAV2, fixed alignment fitted only on
   a disjoint calibration recording, and per-frame oracle alignment.
2. Add a minimum-denominator/conditioning diagnostic to inverse alignment and
   mark unstable frames without concealing them from the all-valid result.
3. Inspect top-error points in RGB/LiDAR overlays and compute boundary versus
   non-boundary metrics before attributing residuals to occlusion or parallax.
4. Evaluate an actual metric model with `--alignment none` as a separate baseline.
5. Store model name, checkpoint, code revision, and complete command in every
   machine-readable result.

## Evidence and reproducibility notes

Evidence inspected:

- `run_depth.py` (model default and `auto` alignment selection)
- `src/depth_models.py` (available model implementations)
- `outputs/recording{1..4}_ZED_B_E1_A_lidar_global_metrics.json`
- `outputs/recording{1..4}_ZED_B_E1_A_evaluation_debug.csv`
- per-frame `outputs/recording*_ZED_B_*_lidar_samples.csv`

The tail statistics were calculated directly from every saved valid point using
`lidar_depth_m` and `absolute_error_m`; no frame or point was removed. These are
post-hoc diagnostics of the existing run, not a new inference run.
