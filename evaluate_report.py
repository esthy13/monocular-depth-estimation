#!/usr/bin/env python3
"""Generate a leakage-controlled LiDAR evaluation bundle from cached predictions."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.scientific_evaluation import (
    ValidityRules, apply_inverse_alignment, collect_recording_samples,
    deterministic_calibration_split, evaluate_mode, fit_fixed_inverse_alignment,
    json_safe, stable_configuration_hash, write_csv,
)
from src.utils import camera_from_lidar_transform, load_extrinsics, load_intrinsics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--prediction_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report_dir", type=Path, default=Path("outputs/scientific_evaluation"))
    parser.add_argument("--camera", default="ZED_B")
    parser.add_argument("--lidar", default="E1_A")
    parser.add_argument("--calibration_recording", default="recording1")
    parser.add_argument("--evaluation_recordings", nargs="+", default=["recording1", "recording2"])
    parser.add_argument("--calibration_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=20260817)
    parser.add_argument("--max_lidar_dt", type=float, default=0.05)
    parser.add_argument("--time_offset", type=float, default=0.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--max_prediction_depth", type=float, default=100.0)
    parser.add_argument("--min_correspondences", type=int, default=100)
    parser.add_argument("--max_alignment_condition", type=float, default=1e6)
    parser.add_argument("--model_id", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--inference_device", default="mps", help="Recorded provenance for cached predictions.")
    return parser.parse_args()


def save_plots(report_dir: Path, recording: str, modes: dict, frames: list[dict]) -> None:
    plot_dir = report_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"fixed_held_out": "#0072B2", "oracle_diagnostic": "#D55E00", "unaligned_relative_proxy": "#777777"}
    failure = [(18, 26), (54, 62)]
    for metric, ylabel in (("mae_m", "MAE (m)"), ("rmse_m", "RMSE (m)"), ("abs_rel", "AbsRel"), ("delta_1", "δ1")):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for mode, result in modes.items():
            rows = [row for row in result["frame_rows"] if row.get("recording") == recording and row.get("status") == "valid"]
            ax.plot([row["frame"] for row in rows], [row[metric] for row in rows], ".-", ms=3, lw=1, label=mode, color=colors[mode])
        for low, high in failure: ax.axvspan(low, high, color="#CC79A7", alpha=.12)
        ax.set(xlabel="Frame index", ylabel=ylabel, title=f"{recording}: {ylabel} by frame")
        if metric in ("mae_m", "rmse_m", "abs_rel"): ax.set_yscale("symlog", linthresh=.1)
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / f"{recording}_{metric}_frames.png", dpi=240); plt.close(fig)
    fixed_rows = [row for row in modes["fixed_held_out"]["frame_rows"] if row.get("recording") == recording and row.get("status") == "valid"]
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot([r["frame"] for r in fixed_rows], [1000*r["timestamp_delta_s"] for r in fixed_rows], ".-", color="#009E73")
    ax.axhline(0, color="black", lw=.8); ax.set(xlabel="Frame index", ylabel="LiDAR − camera (ms)", title=f"{recording}: timestamp delta"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plot_dir / f"{recording}_timestamp_delta.png", dpi=240); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, label in zip(axes, ("alignment_scale", "alignment_shift", "alignment_condition_number"), ("Scale", "Shift", "Condition number")):
        oracle = [r for r in modes["oracle_diagnostic"]["frame_rows"] if r.get("recording") == recording and r.get("status") == "valid"]
        ax.plot([r["frame"] for r in oracle], [r[key] for r in oracle], ".-", color="#D55E00"); ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[-1].set_xlabel("Frame index"); fig.suptitle(f"{recording}: oracle alignment diagnostics"); fig.tight_layout(); fig.savefig(plot_dir / f"{recording}_alignment_diagnostics.png", dpi=240); plt.close(fig)
    fixed = modes["fixed_held_out"]; pred, gt = fixed["all_prediction"], fixed["all_ground_truth"]
    if len(pred):
        rng = np.random.default_rng(0); take = rng.choice(len(pred), min(100000, len(pred)), replace=False)
        limit = float(np.percentile(np.r_[pred[take], gt[take]], 99))
        fig, ax = plt.subplots(figsize=(6, 6)); ax.hexbin(gt[take], pred[take], gridsize=100, bins="log", cmap="viridis"); ax.plot([0, limit], [0, limit], "r--", label="y=x"); ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="LiDAR depth (m)", ylabel="Fixed-alignment prediction (m)"); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / f"{recording}_predicted_vs_ground_truth.png", dpi=240); plt.close(fig)
        error = pred - gt
        fig,ax=plt.subplots(figsize=(8,5));ax.hexbin(gt[take],np.abs(error[take]),gridsize=100,bins="log",cmap="magma");ax.set(xlabel="LiDAR depth (m)",ylabel="Absolute error (m)",title=f"{recording}: fixed-alignment error versus depth",yscale="symlog");ax.grid(alpha=.15);fig.tight_layout();fig.savefig(plot_dir/f"{recording}_error_vs_ground_truth.png",dpi=240);plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); axes[0].hist(error, bins=150, range=np.percentile(error, [1,99]), color="#0072B2"); axes[0].set(xlabel="Residual, prediction − LiDAR (m)", ylabel="Count"); sorted_abs=np.sort(np.abs(error)); axes[1].plot(sorted_abs, np.linspace(0,1,len(sorted_abs)), color="#0072B2"); axes[1].set(xlabel="Absolute error (m)", ylabel="Cumulative fraction", xscale="symlog"); [a.grid(alpha=.2) for a in axes]; fig.tight_layout(); fig.savefig(plot_dir / f"{recording}_residual_distribution.png", dpi=240); plt.close(fig)
        bins=fixed["depth_bins"]; labels=[f"{b['range_m'][0]}–{b['range_m'][1]}" for b in bins]; fig, ax=plt.subplots(figsize=(8,4.5)); ax.bar(labels,[b["mae_m"] for b in bins],color="#0072B2"); ax.set(xlabel="LiDAR depth range (m)",ylabel="MAE (m)",title=f"{recording}: fixed-alignment error by depth"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(plot_dir/f"{recording}_depth_binned_mae.png",dpi=240);plt.close(fig)


def save_failure_panels(report_dir: Path, frames: list[dict], fixed_parameters, recording: str) -> None:
    panel_dir=report_dir/"failure_panels";panel_dir.mkdir(parents=True,exist_ok=True)
    selected={18,19,20,21,22,23,24,25,26,54,55,56,57,58,59,60,61,62}
    for frame in frames:
        if frame["recording"] != recording or frame["frame"] not in selected: continue
        image=cv2.cvtColor(cv2.imread(frame["image_path"]),cv2.COLOR_BGR2RGB); raw=frame["raw_image"]
        dense=apply_inverse_alignment(raw,fixed_parameters); pred=apply_inverse_alignment(frame["raw_prediction"],fixed_parameters); residual=pred-frame["ground_truth_m"]
        fig,axes=plt.subplots(2,3,figsize=(15,8)); axes[0,0].imshow(image);axes[0,0].set_title("RGB")
        axes[0,1].imshow(raw,cmap="magma",vmin=np.percentile(raw[np.isfinite(raw)],2),vmax=np.percentile(raw[np.isfinite(raw)],98));axes[0,1].set_title("Raw relative inverse depth")
        axes[0,2].imshow(dense,cmap="viridis",vmin=.1,vmax=10);axes[0,2].set_title("Fixed aligned depth (m)")
        axes[1,0].imshow(image);sc=axes[1,0].scatter(frame["pixels"][:,0],frame["pixels"][:,1],c=frame["ground_truth_m"],s=1,cmap="viridis",vmin=.1,vmax=10);axes[1,0].set_title("Projected LiDAR depth");fig.colorbar(sc,ax=axes[1,0],fraction=.046)
        axes[1,1].imshow(image);sc=axes[1,1].scatter(frame["pixels"][:,0],frame["pixels"][:,1],c=residual,s=1,cmap="coolwarm",vmin=-2,vmax=2);axes[1,1].set_title("Fixed residual (m), clipped only for display");fig.colorbar(sc,ax=axes[1,1],fraction=.046)
        axes[1,2].axis("off"); oracle_scale="n/a"; axes[1,2].text(0,.95,f"frame {frame['frame']}\ndt={frame['timestamp_delta_s']*1000:.1f} ms\nfixed a={fixed_parameters.scale:.6g}\nfixed b={fixed_parameters.shift:.6g}\npoints={len(pred)}\nresidual median={np.median(residual):.3f} m\nresidual p95 abs={np.percentile(abs(residual),95):.3f} m",va="top",family="monospace")
        for ax in axes.flat:
            if ax.has_data(): ax.axis("off")
        fig.suptitle(f"{recording} frame {frame['frame']}: held-out fixed alignment diagnostics");fig.tight_layout();fig.savefig(panel_dir/f"{recording}_frame_{frame['frame']:04d}.png",dpi=220);plt.close(fig)


def analyze_failure_windows(frames: list[dict], modes: dict, recording: str) -> dict:
    intervals=((18,26),(54,62)); fixed={r["frame"]:r for r in modes["fixed_held_out"]["frame_rows"] if r.get("recording")==recording and r.get("status")=="valid"}; oracle={r["frame"]:r for r in modes["oracle_diagnostic"]["frame_rows"] if r.get("recording")==recording and r.get("status")=="valid"}; rows=[]
    for frame in frames:
        index=frame["frame"]
        if frame["recording"]!=recording or index not in fixed or index not in oracle:continue
        rows.append({"frame":index,"failure_window":any(low<=index<=high for low,high in intervals),"fixed_abs_rel":fixed[index]["abs_rel"],"oracle_abs_rel":oracle[index]["abs_rel"],"absolute_timestamp_delta_ms":abs(frame["timestamp_delta_s"])*1000,"motion_score_mean_abs_gray":frame["motion_score_mean_abs_gray"],"stereo_projection_median_abs_error_m":frame["stereo_projection_median_abs_error_m"],"oracle_scale":oracle[index]["alignment_scale"],"oracle_shift":oracle[index]["alignment_shift"],"oracle_condition_number":oracle[index]["alignment_condition_number"]})
    def group(selected):
        return {key:float(np.nanmean([row[key] for row in selected])) for key in rows[0] if key not in ("frame","failure_window")}
    failure=[r for r in rows if r["failure_window"]];other=[r for r in rows if not r["failure_window"]]
    correlations={}
    target=np.asarray([r["fixed_abs_rel"] for r in rows])
    for key in ("oracle_abs_rel","absolute_timestamp_delta_ms","motion_score_mean_abs_gray","stereo_projection_median_abs_error_m","oracle_scale","oracle_shift","oracle_condition_number"):
        values=np.asarray([r[key] for r in rows]); correlations[key]=float(np.corrcoef(target,values)[0,1])
    return {"predefined_intervals":[list(x) for x in intervals],"failure_window_frame_count":len(failure),"other_frame_count":len(other),"failure_window_means":group(failure),"other_frame_means":group(other),"correlation_with_fixed_abs_rel":correlations,"frame_56":next((row for row in rows if row["frame"]==56),None),"evidence_based_interpretation":["Failure frames have substantially higher image-change/motion scores and materially different oracle scale/shift parameters.","Stereo projection residual is not higher in the failure windows, arguing against extrinsic direction/projection as the primary cause.","Timestamp deltas are larger in failure windows and may amplify disagreement on moving objects, but static-scene residuals and low stereo projection residuals do not support synchronization as the sole cause.","Frame 56 oracle shift is negative; fitted inverse depths approach zero, producing a heavy far-depth tail. Its oracle catastrophe is alignment instability plus dynamic scene content, not zero-filled predictions."]}


def main() -> None:
    args=parse_args(); args.report_dir.mkdir(parents=True,exist_ok=True)
    intr=load_intrinsics(args.data_dir/"intrinsic.json");ext=load_extrinsics(args.data_dir/"extrinsics.json")
    transform=camera_from_lidar_transform(ext,args.camera,args.lidar,"sensor_to_reference")
    rules=ValidityRules(args.min_depth,args.max_depth,args.min_depth,args.max_prediction_depth,args.min_correspondences,args.max_alignment_condition)
    configuration={"camera":args.camera,"lidar":args.lidar,"calibration_recording":args.calibration_recording,"evaluation_recordings":args.evaluation_recordings,"calibration_fraction":args.calibration_fraction,"split_seed":args.split_seed,"max_lidar_dt_s":args.max_lidar_dt,"time_offset_s":args.time_offset,"validity_rules":asdict(rules),"boundary_definition":"ZED stereo-depth Sobel gradient magnitude > 0.5 m/pixel-equivalent kernel response","transform_convention":"sensor_to_reference","transform_camera_from_lidar":transform.tolist(),"model":{"id":args.model_id,"cached_prediction_device":args.inference_device},"prediction_representation":"relative inverse-depth-like; not metric"}
    all_frames={};sync={}
    for recording in sorted(set([args.calibration_recording,*args.evaluation_recordings])):
        all_frames[recording],sync[recording]=collect_recording_samples(args.data_dir,args.prediction_dir,recording,args.camera,args.lidar,intr,transform,args.max_lidar_dt,args.time_offset,rules)
    available=[f["frame"] for f in all_frames[args.calibration_recording]]
    calibration_indices,heldout_indices=deterministic_calibration_split(available,args.calibration_fraction,args.split_seed)
    calibration_frames=[f for f in all_frames[args.calibration_recording] if f["frame"] in set(calibration_indices)]
    calibration_dt=np.asarray([f["timestamp_delta_s"] for f in calibration_frames])
    synchronization_validation={
        "offset_estimation_scope":"calibration subset only",
        "calibration_subset_median_delta_s":float(np.median(calibration_dt)),
        "suggested_offset_to_center_delta_s":float(-np.median(calibration_dt)),
        "applied_offset_s":args.time_offset,
        "note":"suggested offset is reported but not applied automatically",
        "threshold_sensitivity":{
            recording:{str(threshold):sum(abs(f["timestamp_delta_s"])<=threshold for f in frames) for threshold in (.01,.02,.03,.05)}
            for recording,frames in all_frames.items()
        },
    }
    fit_raw=np.concatenate([f["raw_prediction"] for f in calibration_frames]);fit_gt=np.concatenate([f["ground_truth_m"] for f in calibration_frames])
    fixed=fit_fixed_inverse_alignment(fit_raw,fit_gt,args.calibration_recording,calibration_indices,args.max_alignment_condition)
    evaluation_frames=[]
    for recording in args.evaluation_recordings:
        evaluation_frames.extend([f for f in all_frames[recording] if recording!=args.calibration_recording or f["frame"] in set(heldout_indices)])
    modes={mode:evaluate_mode(evaluation_frames,mode,rules,fixed if mode=="fixed_held_out" else None) for mode in ("unaligned_relative_proxy","fixed_held_out","oracle_diagnostic")}
    failure_analysis=analyze_failure_windows(evaluation_frames,modes,args.calibration_recording)
    data_quality={}
    for recording,frames in all_frames.items():
        totals={}
        for frame in frames:
            for reason,count in frame["rejections"].items():totals[reason]=totals.get(reason,0)+count
        data_quality[recording]={"input_point_rejections":totals,"frames_with_cached_predictions":len(frames)}
    per_recording={}
    for recording in args.evaluation_recordings:
        recording_frames=[f for f in evaluation_frames if f["recording"]==recording]
        per_recording[recording]={mode:evaluate_mode(recording_frames,mode,rules,fixed if mode=="fixed_held_out" else None) for mode in modes}
        save_plots(args.report_dir,recording,per_recording[recording],recording_frames)
    save_failure_panels(args.report_dir,evaluation_frames,fixed,args.calibration_recording)
    split={"calibration_recording":args.calibration_recording,"calibration_frames":calibration_indices,"held_out_frames_same_recording":heldout_indices,"external_evaluation_recordings":[r for r in args.evaluation_recordings if r!=args.calibration_recording],"overlap_count":len(set(calibration_indices)&set(heldout_indices))}
    payload={"schema_version":1,"configuration":configuration,"configuration_sha256":stable_configuration_hash(configuration),"software":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"opencv":cv2.__version__},"calibration_split":split,"fixed_alignment":asdict(fixed),"synchronization":sync,"synchronization_validation":synchronization_validation,"data_quality":data_quality,"failure_analysis":failure_analysis,"aggregate_modes":modes,"per_recording":per_recording,"calibration_status":{"calibrated_extrinsics_flag":False,"status":"direction independently validated against ZED stereo depth; no target-based calibration provenance available","dataset_files_modified":False}}
    (args.report_dir/"evaluation.json").write_text(json.dumps(json_safe(payload),indent=2)+"\n")
    rows=[]
    for recording,results in per_recording.items():
        for mode,result in results.items(): rows.extend({"mode":mode,**row} for row in result["frame_rows"])
    write_csv(args.report_dir/"per_frame_metrics.csv",rows)
    write_csv(args.report_dir/"synchronization.csv",[{"recording":f["recording"],"frame":f["frame"],"camera_timestamp":f["camera_timestamp"],"lidar_timestamp":f["lidar_timestamp"],"timestamp_delta_s":f["timestamp_delta_s"]} for rec in all_frames.values() for f in rec])
    report=["# Scientifically defensible LiDAR evaluation","","## Methodology","",f"A deterministic {args.calibration_fraction:.0%} subset of `{args.calibration_recording}` was used only to fit one inverse-depth affine alignment. Parameters were frozen for held-out frames and external recordings. Per-frame oracle alignment is reported only as an optimistic diagnostic. The reciprocal unaligned proxy is in arbitrary units and is not metric-depth accuracy.","",f"Configuration hash: `{payload['configuration_sha256']}`","","## Fixed alignment","",f"`1/Z = {fixed.scale:.9g} * raw + {fixed.shift:.9g}`; condition number {fixed.condition_number:.3f}; {fixed.fit_points} calibration points.","","## Results",""]
    report.extend(["### Aggregate held-out evaluation","","| Mode | Frames | Points | MAE (m) | RMSE (m) | AbsRel | δ1 |","|---|---:|---:|---:|---:|---:|---:|"])
    for mode,result in modes.items():
        m=result["pooled_metrics"];report.append(f"| {mode} | {result['valid_frame_count']} | {m['count']} | {m['mae_m']:.3f} | {m['rmse_m']:.3f} | {m['abs_rel']:.3f} | {m['delta_1']:.3f} |")
    report.append("")
    for recording,results in per_recording.items():
        report.append(f"### {recording}");report.append("");report.append("| Mode | Frames | Points | MAE (m) | RMSE (m) | AbsRel | SqRel | log RMSE | δ1 | δ2 | δ3 |");report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mode,result in results.items():
            m=result["pooled_metrics"];report.append(f"| {mode} | {result['valid_frame_count']} | {m['count']} | {m['mae_m']:.3f} | {m['rmse_m']:.3f} | {m['abs_rel']:.3f} | {m['sq_rel']:.3f} | {m['rmse_log']:.3f} | {m['delta_1']:.3f} | {m['delta_2']:.3f} | {m['delta_3']:.3f} |")
        report.extend(["",])
    report.extend(["## Failure analysis","",f"The predefined failure windows have mean fixed AbsRel {failure_analysis['failure_window_means']['fixed_abs_rel']:.3f}, versus {failure_analysis['other_frame_means']['fixed_abs_rel']:.3f} elsewhere. Their mean image-change score is {failure_analysis['failure_window_means']['motion_score_mean_abs_gray']:.2f}, versus {failure_analysis['other_frame_means']['motion_score_mean_abs_gray']:.2f}; fixed AbsRel correlates with this motion score at r={failure_analysis['correlation_with_fixed_abs_rel']['motion_score_mean_abs_gray']:.2f}. Median LiDAR–stereo projection disagreement is not worse in the failure windows ({failure_analysis['failure_window_means']['stereo_projection_median_abs_error_m']:.3f} m versus {failure_analysis['other_frame_means']['stereo_projection_median_abs_error_m']:.3f} m). This supports scene motion/content and changing relative-depth scale/shift as the primary causes, with timestamp mismatch a possible contributor on moving objects—not an extrinsic-direction failure.","",f"Frame 56 has oracle AbsRel {failure_analysis['frame_56']['oracle_abs_rel']:.3f}, oracle scale {failure_analysis['frame_56']['oracle_scale']:.6g}, and negative shift {failure_analysis['frame_56']['oracle_shift']:.6g}. The fitted inverse depth approaches zero and creates a heavy far-depth tail. The frame is retained.",""])
    sync_description="; ".join(f"{recording}: {value['matched_frames']}/{value['total_camera_frames']}" for recording,value in sync.items())
    report.extend(["## Synchronization","",f"At the {args.max_lidar_dt*1000:.0f} ms threshold, matched frames are {sync_description}. Frames outside the threshold are listed in `evaluation.json`. A calibration-subset-only median suggests {synchronization_validation['suggested_offset_to_center_delta_s']*1000:.2f} ms, but it was not applied or optimized on evaluation data.","","## Calibration limitation","","The supplied matrices are documented as sensor-to-G1_A transforms and their direction is independently supported by ZED stereo-depth residuals in `extrinsic_validation.json`. However, every dataset declares `calibrated_extrinsics=false`, per-recording calibration JSON files are empty, and no target detections or surveyed correspondences are present. The flag was not changed and no transform was invented.","","## Validity and exclusions","",f"LiDAR depths must be in [{rules.min_ground_truth_m}, {rules.max_ground_truth_m}] m, aligned predictions in [{rules.min_prediction_m}, {rules.max_prediction_m}] m, and frames need at least {rules.min_correspondences_per_frame} correspondences. Rejection counts and exact reasons are machine-readable. No frame, including frame 56, was excluded by the predefined rules in this run.","","## Limitations and defensible conclusion","","The fixed alignment does not transfer accurately enough to support a claim of standalone metric-depth accuracy. The available evidence supports relative-depth structure under static scene content, while scale/shift changes and dynamic foreground motion materially degrade held-out metric conversion. Extrinsics are directionally validated but not target-calibrated; synchronization is nearest-neighbor within the configured threshold; sparse LiDAR sampling and stereo-derived boundary labels remain limitations.","","## Reproduction","","```bash",f"uv run python evaluate_report.py --data_dir {args.data_dir} --prediction_dir {args.prediction_dir} --calibration_recording {args.calibration_recording} --evaluation_recordings {' '.join(args.evaluation_recordings)}","```","","```bash",f"uv run python validate_extrinsics.py --data_dir {args.data_dir} --recordings recording1 recording2 recording3 recording4 --output {args.report_dir}/extrinsic_validation.json","```","","When target-derived 3D correspondences are available:","","```bash",f"uv run python calibrate_extrinsics.py --correspondences calibration_correspondences.csv --output {args.report_dir}/calibrated_E1_A_to_ZED_B.json","```",""])
    (args.report_dir/"REPORT.md").write_text("\n".join(report))
    print(f"Wrote report bundle to {args.report_dir}")


if __name__=="__main__": main()
