## Comparison: Depth Anything V2 vs LiDAR (per-frame oracle alignment)

> Correction (2026-08-18): this run used the default Depth Anything V2 model,
> not UniDepth V2. The reported results use `inverse_least_squares` fitted
> independently on each frame with the same LiDAR samples that are scored.
> They measure oracle-aligned relative-depth shape and must not be presented as
> metric-depth accuracy.

#### Recording 1
Timestamp dt (LiDAR + offset - camera): min=-0.028497, max=0.038958, mean=0.005241, median=0.005174, std=0.025265 s
Global LiDAR metrics (530505 points across 66 frames): MAE=0.614 m, RMSE=3.183 m, AbsRel=0.158

#### Recording 2
Timestamp dt (LiDAR + offset - camera): min=-0.026847, max=0.040564, mean=0.007355, median=0.006928, std=0.026308 s
Global LiDAR metrics (530382 points across 65 frames): MAE=0.305 m, RMSE=0.850 m, AbsRel=0.066

#### Recording 3
Timestamp dt (LiDAR + offset - camera): min=-0.032275, max=0.045993, mean=0.010397, median=0.010723, std=0.027212 s
Global LiDAR metrics (9550732 points across 1182 frames): MAE=0.340 m, RMSE=2.403 m, AbsRel=0.085

#### Recording 4
Timestamp dt (LiDAR + offset - camera): min=-0.043560, max=0.033057, mean=-0.010068, median=-0.009693, std=0.026864 s
Global LiDAR metrics (1651617 points across 201 frames): MAE=0.322 m, RMSE=0.831 m, AbsRel=0.075
