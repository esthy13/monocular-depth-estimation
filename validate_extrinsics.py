#!/usr/bin/env python3
"""Validate supplied extrinsic direction against synchronized ZED stereo depth."""
from __future__ import annotations

import argparse,json
from pathlib import Path
import cv2
import numpy as np
from src.lidar_evaluation import keep_nearest_per_pixel,load_lidar_points,project_perspective_points,sample_bilinear,save_projection_overlay,transform_points
from src.utils import camera_from_lidar_transform,find_rgb_images,find_sensor_npy_files,load_extrinsics,load_intrinsics,match_by_timestamp


def main():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter);p.add_argument("--data_dir",type=Path,required=True);p.add_argument("--recordings",nargs="+",default=["recording1","recording2","recording3","recording4"]);p.add_argument("--camera",default="ZED_B");p.add_argument("--lidar",default="E1_A");p.add_argument("--max_lidar_dt",type=float,default=.05);p.add_argument("--max_frames_per_recording",type=int,default=100);p.add_argument("--output",type=Path,default=Path("outputs/extrinsic_validation.json"));args=p.parse_args()
    intr=load_intrinsics(args.data_dir/"intrinsic.json");ext=load_extrinsics(args.data_dir/"extrinsics.json");results={}
    for convention in ("sensor_to_reference","reference_to_sensor"):
        transform=camera_from_lidar_transform(ext,args.camera,args.lidar,convention);recordings={}
        for recording in args.recordings:
            images=find_rgb_images(args.data_dir,args.camera,recording);lidars=find_sensor_npy_files(args.data_dir,args.lidar,recording);depths=find_sensor_npy_files(args.data_dir,f"{args.camera}_depth",recording);lp=dict(match_by_timestamp(images,lidars,args.max_lidar_dt));dp=dict(match_by_timestamp(images,depths,.001));step=max(1,len(images)//args.max_frames_per_recording);errors=[];signed=[];rel=[]
            for image in images[::step]:
                if image not in lp or image not in dp:continue
                points=transform_points(load_lidar_points(lp[image]),transform);pixels,z,_=project_perspective_points(points,intr[args.camera],(720,1280));nearest=keep_nearest_per_pixel(pixels,z);pixels,z=pixels[nearest],z[nearest];stereo=np.load(dp[image]).squeeze();sample=sample_bilinear(stereo,pixels);valid=np.isfinite(sample)&(sample>.1)&(sample<19.5)&np.isfinite(z)&(z>.1)&(z<19.5);delta=z[valid]-sample[valid];errors.extend(abs(delta));signed.extend(delta);rel.extend(abs(delta)/sample[valid])
            recordings[recording]={"points":len(errors),"mae_m":float(np.mean(errors)),"median_absolute_error_m":float(np.median(errors)),"p90_absolute_error_m":float(np.percentile(errors,90)),"p95_absolute_error_m":float(np.percentile(errors,95)),"mean_signed_error_m":float(np.mean(signed)),"abs_rel":float(np.mean(rel))}
        results[convention]={"transform_camera_from_lidar":transform.tolist(),"recordings":recordings}
    overlay_dir=args.output.parent/"extrinsic_overlays";overlay_dir.mkdir(parents=True,exist_ok=True);preferred=camera_from_lidar_transform(ext,args.camera,args.lidar,"sensor_to_reference")
    representative=[]
    for recording in args.recordings[:2]:
        images=find_rgb_images(args.data_dir,args.camera,recording);lidars=find_sensor_npy_files(args.data_dir,args.lidar,recording);pairs=dict(match_by_timestamp(images,lidars,args.max_lidar_dt))
        requested=[2,28,56,63] if recording=="recording1" else [2,25,45,63]
        for frame in requested:
            if frame>=len(images) or images[frame] not in pairs:continue
            image=cv2.imread(str(images[frame]));points=transform_points(load_lidar_points(pairs[images[frame]]),preferred);pixels,z,_=project_perspective_points(points,intr[args.camera],image.shape[:2]);nearest=keep_nearest_per_pixel(pixels,z);path=overlay_dir/f"{recording}_frame_{frame:04d}.png";save_projection_overlay(image,pixels[nearest],z[nearest],path);representative.append(str(path))
    payload={"status":"validation_only_not_target_calibration","reference":"synchronized ZED_B stereo depth, capped below 19.5 m","results":results,"representative_overlays":representative,"conclusion":"sensor_to_reference direction is supported when its residuals are substantially lower; dataset calibrated_extrinsics flags remain unchanged","limitations":["stereo depth is not a surveyed calibration target","dynamic objects and occlusions contribute residuals","no target detections or authoritative covariance supplied"]};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+"\n");print(args.output)
if __name__=="__main__":main()
