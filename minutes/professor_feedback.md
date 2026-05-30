# 20.05.2026
Hey guys, i tried to calibrate the setup after our recording, but i was unable to do so. However, i now recorded some new recordings of myself moving around. I also have the entire intrinsic and extrinsic calibration parameters.

You can find the data under this link (the password is "password"): https://uni-bielefeld.sciebo.de/s/9MQiScN548t3RCa
It contains:
- intrinsics.json -> Camera intrinsics of both cameras (fisheye+perspective) following the opencv conention of camera matrix K and distortion coefficients d
- extrinsics.json -> Homogenous transformation matrices for all sensors describing their pose relative to the main reference sensor "G1_A" (the fisheye camera)
- recordings 1-4 -> The recorded data for each sensor is under data/. To load sensor data from the same time, compare the filenames containing the system timestamp. The camera rgb images are stored as jpegs, the point clouds (from the lidars) and the depth images (from the stereocamera) are stored as numpy arrays.

To get started, i suggest you setup a python script to run a monocular depth estimation model on your machine. Then, you load images from our recording (or any other image you want to play around with) and visualize the resulting depth image. 

- To visualize images (and also do more complex computer vision stuff) you can use opencv: https://opencv.org/
- To visualize point clouds you can use open3d: https://www.open3d.org/

Let me know if something doesnt work for you! If anything remains unclear, feel free to ask :)