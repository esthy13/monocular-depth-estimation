# 30.05.2026
## idea
1. Build a small data loader for the recordings 1–4 that matches RGB images by timestamp.
2. Run a pretrained monocular depth model on one image.
3. Convert the raw output into a depth map and display it with OpenCV.
4. Compare the predicted depth with the stereo or lidar data after alignment.
5. Only after that consider fine-tuning if the zero-shot result is not good enough.

### Model's features:
UniDepthV2 if you want the simplest strong metric-depth baseline.
Depth Any Camera if the fisheye camera is central, because it is designed for arbitrary camera types.
Metric3Dv2 if you want a more geometry-focused foundation model.

# What we found out:
1. UniDepthV2 works out very well with perspective projection camera
2. UniDepthV2 and Depth Any Camera don't work well with the fisheye camera (the border of the camera is detected as an element near the camera, and the rest is detected as very far away, and in the second case nothing is really detected) 
3. Depth Any Camera does not work so well with perspective projection cameras like UniDepthV2 did. 

Possible solution to current issues: https://github.com/guoyangzhao/FisheyeDepth
they implement a depth model only for fisheye cameras.