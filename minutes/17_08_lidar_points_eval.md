# Output meaning:

```bash
(monocular-depth-estimation) esther@mac monocular-depth-estimation % uv run python run_depth.py \
    --data_dir cv_project_data \
    --sensor ZED_B \
    --recording recording1 \
    --image_index 1 \
    --evaluate_lidar \
    --lidar_sensor E1_A

Intrinsics loaded for: ['ZED_B', 'G1_A']
Extrinsics loaded for: ['ZED_B', 'G1_A', 'E1_A', 'E1_B']
Camera model: perspective  |  fx=530.0  fy=528.9  cx=640.7  cy=385.4
Found 68 images for 'ZED_B' in 'recording1'
Image: 0000000001_1779291267.098958254.jpg
Loaded image: 1280×720 px
Loading depth-anything/Depth-Anything-V2-Small-hf on mps ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
preprocessor_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 775/775 [00:00<00:00, 1.73MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 950/950 [00:00<00:00, 2.64MB/s]
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 99.2M/99.2M [00:42<00:00, 2.32MB/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 287/287 [00:00<00:00, 19431.24it/s]
Model loaded.
Depth map: 1280×720 px  raw range [-0.034, 6.195]
Saved RGB image:      outputs/recording1_ZED_B_0001_rgb.jpg
Saved visualization:  outputs/recording1_ZED_B_0001_depth.png
Saved raw depth:      outputs/recording1_ZED_B_0001_depth_raw.npy
LiDAR cloud: 0000000012_1779291267.106816769.npy
Saved LiDAR projection: outputs/recording1_ZED_B_0001_lidar_projection.png (8193 visible points)
LiDAR metrics (8188 points): MAE=2.025 m, RMSE=5.414 m, AbsRel=0.709
Saved metrics:        outputs/recording1_ZED_B_0001_lidar_metrics.json
Saved point samples:  outputs/recording1_ZED_B_0001_lidar_samples.csv
```

This output shows that the Python script successfully executed a **monocular depth estimation** analysis. It took a single camera image, computed an AI-based depth map, and compared it against ground-truth LiDAR sensor data to evaluate its accuracy. 

Here is the detailed breakdown grouped by logical steps: 

### 1. Initialization and Configuration

* **Calibration Loading**: The system loaded the camera's internal parameters (**Intrinsics**) and the relative positions between sensors (**Extrinsics**).
* **Camera Model**: It uses a *perspective* model with a resolution of 1280 × 720 pixels (deducted from the cx/cy optical centers and the saved files).
* **Data**: It found 68 images total and selected the first one from the list (0000000001_...jpg).

### 2. AI Model Details

* **Model**: It downloaded the **Depth-Anything-V2-Small-hf** model from Hugging Face (around 99 MB).
* **Hardware**: It loaded it onto **mps** (Metal Performance Shaders), meaning it is using your Mac's GPU acceleration.
* **Predicted Output**: It generated a raw depth map with values ranging from -0.034 to 6.195 (relative/inverse values typical for monocular models before scale alignment).

### 3. Saved Output Files

The script created four main files in the outputs/ folder: 

* ..._rgb.jpg: The original image extracted from the sequence.
* ..._depth.png: A colorized visualization of the depth map (e.g., closer objects in red, farther in blue).
* ..._depth_raw.npy: The precise numerical data of the depth map in NumPy format.
* ..._lidar_projection.png: An image showing **8,193 LiDAR points** projected and overlaid onto the camera image.

### 4. Evaluation and LiDAR Metrics (The Key Result)

The script grabbed the chronologically matching LiDAR file (0000000012_...npy), aligned the point cloud, and calculated the AI model's error by comparing its predictions to the LiDAR ground truth (across 8,188 valid points): 

* **MAE (Mean Absolute Error) = 2.025 m**: On average, the depth estimated by the AI misses the actual distance measured by the LiDAR by **2 meters and 2 centimeters**.
* **RMSE (Root Mean Squared Error) = 5.414 m**: The root mean squared error. Because it is significantly higher than the MAE, it indicates the presence of severe **macroscopic errors** (outliers) on certain points (e.g., the script estimated an object at 5 meters when it was actually 20 meters away).
* **AbsRel (Absolute Relative Error) = 0.709**: The average relative error is **70.9%**. This is a very high value, which typically means the monocular model's relative depth has not been properly scaled or calibrated to the actual metric scale of the LiDAR yet.

# Changelog:
1. I should try adding **a scale alignment step (like Least Squares scaling) between the AI map and the LiDAR points before calculating metrics**. This could reduce the error, that right now is too high --> implemented, the absolute error and the MAE are consistently reduced

2. I implemented a batch method to evaluate all images, note that 66 of 68 RGB frames have an E1_A match within 50 ms (the images that do not match within 50 ms with the lidar sensor are going to be excluded, I think) --> absolute error is too big

3.  Implemented and verified. The systematic MAE≈4.5 m / AbsRel≈1.0 pattern was caused by incorrect relative-depth alignment, not Depth Anything V2 accuracy.

  ### Confirmed root cause

  Depth Anything V2 Small is a relative-depth model; its raw output is inverse-depth/disparity-like, not metres (official
  model card (https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), Transformers documentation
  (https://huggingface.co/docs/transformers/main/model_doc/depth_anything_v2)).

  The previous pipeline performed:

  distance_proxy = 1 / raw_prediction
  depth_metric = scale * distance_proxy

  This is mathematically insufficient because relative inverse depth has both scale and shift ambiguity. Raw values close
  to zero became enormous after reciprocal conversion, driving the least-squares scale to approximately 10⁻⁹. Almost every
  resulting metric prediction was therefore near zero:

  abs(pred - gt) / gt ≈ abs(0 - gt) / gt ≈ 1

  That directly explains the repeated AbsRel≈1 and MAE near the dataset’s mean LiDAR depth.

  The corrected formulation is:

  inverse_depth_metric = a * raw_prediction + b
  depth_metric = 1 / inverse_depth_metric

  Only finite, positive fitted inverse depths are evaluated.

  ### Changes

  - Corrected and hardened inverse-depth affine alignment in src/lidar_evaluation.py:114.
  - Made --alignment auto explicitly select per-frame inverse-depth alignment for Depth Anything V2 in run_depth.py:285.
  - Prevented recording-wide metrics from silently fitting a second, semantically different alignment. Global metrics now
    aggregate already per-frame-aligned predictions in run_depth.py:392.

  - Added explicit validation for NaN, Inf, zero, negative, degenerate fits, nonpositive scale, and nonpositive aligned
    inverse depth.

  - Added separate invalid LiDAR, invalid sampled prediction, and post-alignment exclusion counts.
  - Added full per-frame diagnostic CSV output, camera-coordinate ranges, timestamp statistics, prediction distributions,
    alignment parameters, and 20 reproducible samples for frames with AbsRel≥0.9.

  - Added --time_offset, without changing the default timestamp tolerance or applying an unsupported offset.
  - Added four-panel RGB/LiDAR/prediction overlays and recording-prefixed evaluation plots.
  - Point CSVs now include absolute and relative errors.
  - Added alignment and timestamp-offset tests.

