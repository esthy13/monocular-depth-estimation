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

# Future work:
I should try adding **a scale alignment step (like Least Squares scaling) between the AI map and the LiDAR points before calculating metrics**. This could reduce the error, that right now is too high