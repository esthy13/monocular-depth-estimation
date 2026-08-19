#!/usr/bin/env python3
"""Estimate a LiDAR-to-camera rigid transform from supplied 3D correspondences.

The input CSV must contain lidar_x, lidar_y, lidar_z, camera_x, camera_y,
camera_z. This command deliberately cannot infer correspondences from monocular
predictions or evaluation errors.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 3 or np.linalg.matrix_rank(source - source.mean(0)) < 2:
        raise ValueError("at least three non-collinear 3D correspondences are required")
    source_center, target_center = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-source_center).T @ (target-target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1; rotation = vt.T @ u.T
    transform = np.eye(4); transform[:3,:3] = rotation
    transform[:3,3] = target_center - rotation @ source_center
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3,:3].T + transform[:3,3]


def ransac_rigid(source: np.ndarray, target: np.ndarray, threshold_m: float, iterations: int, seed: int):
    rng=np.random.default_rng(seed);best=np.zeros(len(source),dtype=bool)
    for _ in range(iterations):
        chosen=rng.choice(len(source),3,replace=False)
        try: candidate=rigid_transform(source[chosen],target[chosen])
        except ValueError: continue
        inliers=np.linalg.norm(transform_points(source,candidate)-target,axis=1)<=threshold_m
        if inliers.sum()>best.sum(): best=inliers
    if best.sum()<3: raise ValueError("RANSAC found fewer than three inliers")
    return rigid_transform(source[best],target[best]),best


def main():
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--correspondences",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("outputs/calibrated_E1_A_to_ZED_B.json"))
    parser.add_argument("--source_sensor",default="E1_A");parser.add_argument("--target_sensor",default="ZED_B")
    parser.add_argument("--ransac_threshold_m",type=float,default=.05);parser.add_argument("--iterations",type=int,default=5000);parser.add_argument("--seed",type=int,default=20260817)
    args=parser.parse_args();rows=list(csv.DictReader(args.correspondences.open()))
    required=["lidar_x","lidar_y","lidar_z","camera_x","camera_y","camera_z"]
    missing=[key for key in required if not rows or key not in rows[0]]
    if missing: raise ValueError(f"correspondence CSV is missing columns: {missing}")
    values=np.asarray([[float(row[key]) for key in required] for row in rows]);finite=np.isfinite(values).all(1);values=values[finite]
    transform,inliers=ransac_rigid(values[:,:3],values[:,3:],args.ransac_threshold_m,args.iterations,args.seed)
    residual=np.linalg.norm(transform_points(values[:,:3],transform)-values[:,3:],axis=1)
    payload={"status":"estimated_not_automatically_promoted","source_sensor":args.source_sensor,"target_sensor":args.target_sensor,"convention":"P_target = T_target_source @ P_source","transform":transform.tolist(),"provenance":{"input_csv":str(args.correspondences.resolve()),"created_utc":datetime.now(timezone.utc).isoformat(),"algorithm":"3-point RANSAC followed by Kabsch/SVD on inliers","threshold_m":args.ransac_threshold_m,"iterations":args.iterations,"seed":args.seed,"input_rows":len(rows),"finite_rows":len(values),"inliers":int(inliers.sum())},"residual_m":{"mean":float(residual[inliers].mean()),"median":float(np.median(residual[inliers])),"rmse":float(np.sqrt(np.mean(residual[inliers]**2))),"p95":float(np.percentile(residual[inliers],95)),"max":float(residual[inliers].max())},"validation_required":"independent held-out target poses/correspondences before setting calibrated_extrinsics=true"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+"\n");print(args.output)


if __name__=="__main__":main()
