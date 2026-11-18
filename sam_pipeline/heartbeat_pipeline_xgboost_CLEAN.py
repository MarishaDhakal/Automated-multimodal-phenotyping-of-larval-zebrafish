#!/usr/bin/env python3

import os
import csv
import math
import json
import shutil
import atexit
import cProfile
import pstats
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd


_AUTO_PROFILE_STATE = {
    "profiler": None,
    "prof_path": None,
    "txt_path": None,
    "tsv_path": None,
    "script_basename": "",
    "well": "",
    "top_n": 200,
}


def _write_script_profile_tsv(stats_obj, script_basename, out_tsv, total_exec_sec, main_cum_sec):
    rows = []

    for (filename, line, funcname), (cc, nc, tt, ct, _callers) in stats_obj.stats.items():
        if script_basename in filename:
            pct_main = (ct / main_cum_sec * 100.0) if np.isfinite(main_cum_sec) and main_cum_sec > 0 else np.nan
            pct_total = (ct / total_exec_sec * 100.0) if np.isfinite(total_exec_sec) and total_exec_sec > 0 else np.nan
            rows.append({
                "func": f"{Path(filename).name}:{line}::{funcname}",
                "cum_sec": float(ct),
                "tot_sec": float(tt),
                "total_calls": int(nc),
                "primitive_calls": int(cc),
                "pct_of_main": float(pct_main) if np.isfinite(pct_main) else np.nan,
                "pct_of_total_exec": float(pct_total) if np.isfinite(pct_total) else np.nan,
            })

    rows = sorted(rows, key=lambda r: r["cum_sec"], reverse=True)

    with open(out_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "func",
            "cum_sec",
            "tot_sec",
            "total_calls",
            "primitive_calls",
            "pct_of_main",
            "pct_of_total_exec",
        ])
        for row in rows:
            writer.writerow([
                row["func"],
                f"{row['cum_sec']:.6f}",
                f"{row['tot_sec']:.6f}",
                row["total_calls"],
                row["primitive_calls"],
                f"{row['pct_of_main']:.2f}" if np.isfinite(row["pct_of_main"]) else "nan",
                f"{row['pct_of_total_exec']:.2f}" if np.isfinite(row["pct_of_total_exec"]) else "nan",
            ])


def _finalize_auto_profile():
    global _AUTO_PROFILE_STATE

    profiler = _AUTO_PROFILE_STATE.get("profiler")
    if profiler is None:
        return

    _AUTO_PROFILE_STATE["profiler"] = None

    prof_path = Path(_AUTO_PROFILE_STATE["prof_path"])
    txt_path = Path(_AUTO_PROFILE_STATE["txt_path"])
    tsv_path = Path(_AUTO_PROFILE_STATE["tsv_path"])
    script_basename = str(_AUTO_PROFILE_STATE.get("script_basename", ""))
    well = str(_AUTO_PROFILE_STATE.get("well", ""))
    top_n = int(_AUTO_PROFILE_STATE.get("top_n", 200))

    try:
        profiler.disable()
    except Exception:
        pass

    ensure_dir(prof_path.parent)

    profiler.dump_stats(str(prof_path))
    stats_obj = pstats.Stats(profiler)

    total_exec_sec = np.nan
    main_cum_sec = np.nan

    for (filename, _line, funcname), (_cc, _nc, _tt, ct, _callers) in stats_obj.stats.items():
        if filename == "~" and funcname == "<built-in method builtins.exec>":
            total_exec_sec = float(ct)
            break

    for (filename, _line, funcname), (_cc, _nc, _tt, ct, _callers) in stats_obj.stats.items():
        if script_basename in filename and funcname == "main":
            main_cum_sec = float(ct)
            break

    with open(txt_path, "w") as f:
        f.write(f"well\t{well}\n")
        f.write(f"profile_binary\t{prof_path}\n")
        f.write(f"total_exec_sec\t{total_exec_sec:.6f}\n" if np.isfinite(total_exec_sec) else "total_exec_sec\tnan\n")
        f.write(f"main_cum_sec\t{main_cum_sec:.6f}\n" if np.isfinite(main_cum_sec) else "main_cum_sec\tnan\n")
        f.write("\nTOP_FUNCTIONS_BY_CUMULATIVE_TIME\n")

        stats_cum = pstats.Stats(profiler, stream=f)
        stats_cum.sort_stats("cumulative")
        stats_cum.print_stats(top_n)

        f.write("\nTOP_FUNCTIONS_BY_SELF_TIME\n")
        stats_tot = pstats.Stats(profiler, stream=f)
        stats_tot.sort_stats("tottime")
        stats_tot.print_stats(top_n)

    _write_script_profile_tsv(
        stats_obj=stats_obj,
        script_basename=script_basename,
        out_tsv=tsv_path,
        total_exec_sec=total_exec_sec,
        main_cum_sec=main_cum_sec,
    )

    print(f"[PROFILE] Saved cProfile binary: {prof_path}")
    print(f"[PROFILE] Saved cProfile report: {txt_path}")
    print(f"[PROFILE] Saved script-only table: {tsv_path}")


def _start_auto_profile(out_dir, well, enabled=True, top_n=200):
    global _AUTO_PROFILE_STATE

    if not enabled:
        return

    profile_dir = Path(out_dir) / "profiling"
    ensure_dir(profile_dir)

    run_tag = f"well_{well}"
    prof_path = profile_dir / f"{run_tag}.cprofile.prof"
    txt_path = profile_dir / f"{run_tag}.cprofile.txt"
    tsv_path = profile_dir / f"{run_tag}.function_times.tsv"

    profiler = cProfile.Profile()

    _AUTO_PROFILE_STATE = {
        "profiler": profiler,
        "prof_path": str(prof_path),
        "txt_path": str(txt_path),
        "tsv_path": str(tsv_path),
        "script_basename": Path(__file__).name,
        "well": str(well),
        "top_n": int(top_n),
    }

    atexit.register(_finalize_auto_profile)
    profiler.enable()

    print(
        f"[PROFILE] Enabled automatic cProfile for well {well}. "
        f"Outputs will be written to {profile_dir}"
    )



# =========================================================
# XGBoost fallback defaults
# =========================================================
# These defaults are used automatically by the pipeline.
# The launcher / sbatch file does NOT need to pass model settings.
RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XGB_MODEL_PATH = str(RELEASE_ROOT / "models" / "alive_weak_strong_xgb.json")
DEFAULT_FALLBACK_MODEL_MODE = "collapsed"
DEFAULT_XGB_ALIVE_THRESHOLD = 0.50


# =========================================================
# Shared pipeline helpers
# =========================================================
from heartbeat_utils import (
    ensure_dir,
    find_valid_mask_by_search,
    union_bbox_from_masks,
    write_results_csv,
)


# =========================================================
# Circle / ring detection helpers
# =========================================================
def fit_circle_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        raise RuntimeError(f"Too few pixels for fit: {len(xs)}")

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    cx, cy, c0 = c
    r = np.sqrt(max(c0 + cx * cx + cy * cy, 0.0))
    return float(cx), float(cy), float(r)


def circle_fit_rmse(mask, cx, cy, r):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.inf
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return float(np.sqrt(np.mean((d - r) ** 2)))


def angular_coverage(mask, center_xy=None, nbins=180):
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return 0.0, 0.0

    if center_xy is None:
        cx0 = xs.mean()
        cy0 = ys.mean()
    else:
        cx0, cy0 = center_xy

    ang = np.arctan2(ys - cy0, xs - cx0)
    ang = (ang + 2 * np.pi) % (2 * np.pi)

    bins = np.linspace(0.0, 2 * np.pi, nbins + 1)
    hist, _ = np.histogram(ang, bins=bins)
    occupied = hist > 0
    frac = occupied.mean()
    coverage_rad = frac * 2 * np.pi
    return float(coverage_rad), float(frac)


def make_thin_ring_from_component(comp_u8, inner_band_low=2.0, inner_band_high=6.0):
    dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
    ring = (dist >= inner_band_low) & (dist <= inner_band_high)
    return ring


def detect_ring_auto_prefer_annulus(
    rgb,
    p_range=range(6, 16),
    morph_ksize=7,
    min_area=100000,
    max_fill_ratio=0.75,
    min_ring_pixels=200,
    min_coverage_rad=1.5 * np.pi,
    inner_band_low=2.0,
    inner_band_high=6.0,
    expected_center=None,
    center_weight=0.002,
):
    # Hardcoded minimum acceptable ring radius
    MIN_RADIUS = 400

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {rgb.shape}")

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    if expected_center is None:
        expected_center = (W / 2.0, H / 2.0)

    if morph_ksize % 2 == 0:
        morph_ksize += 1

    k_morph = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
    )

    best = None

    for p_dark in p_range:
        thr = float(np.percentile(gray, p_dark))
        binary = (gray <= thr).astype(np.uint8) * 255

        binary2 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_morph)
        binary2 = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, k_morph)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (binary2 > 0).astype(np.uint8), connectivity=8
        )
        if n <= 1:
            continue

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            comp = (labels == i).astype(np.uint8)

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            fill_ratio = area / (w * h + 1e-6)
            if fill_ratio > max_fill_ratio:
                continue

            ring = make_thin_ring_from_component(
                comp,
                inner_band_low=inner_band_low,
                inner_band_high=inner_band_high,
            )

            ring_pixels = int(ring.sum())
            if ring_pixels < min_ring_pixels:
                continue

            try:
                cx, cy, radius = fit_circle_from_mask(ring)

                # Reject tiny false rings
                if radius < MIN_RADIUS:
                    continue

                rmse = circle_fit_rmse(ring, cx, cy, radius)

            except Exception:
                continue

            coverage_rad, coverage_frac = angular_coverage(
                ring, center_xy=(cx, cy), nbins=180
            )
            if coverage_rad < min_coverage_rad:
                continue

            center_dist = float(np.hypot(cx - expected_center[0], cy - expected_center[1]))

            score = (
                2.5 * coverage_frac
                + 1.2 * np.log(area + 1.0)
                - 0.20 * rmse
                - center_weight * center_dist
            )

            item = {
                "score": float(score),
                "p_dark": int(p_dark),
                "thr": float(thr),
                "component_mask": (comp > 0),
                "ring_mask": ring,
                "cx": float(cx),
                "cy": float(cy),
                "radius": float(radius),
                "rmse": float(rmse),
                "area": int(area),
                "fill_ratio": float(fill_ratio),
                "ring_pixels": int(ring_pixels),
                "coverage_rad": float(coverage_rad),
                "coverage_frac": float(coverage_frac),
                "center_dist": float(center_dist),
                "bbox": (x, y, x + w, y + h),
            }

            if best is None or item["score"] > best["score"]:
                best = item

    if best is None:
        raise RuntimeError("Could not detect a good annular ring candidate")

    return best


# =========================================================
# Crop helpers
# =========================================================
def center_crop_origin_from_detected_center(cx, cy, frame_w, frame_h, crop_size=1048):
    cx_i = int(round(cx))
    cy_i = int(round(cy))

    xoff = cx_i - crop_size // 2
    yoff = cy_i - crop_size // 2

    xoff = max(0, min(xoff, frame_w - crop_size))
    yoff = max(0, min(yoff, frame_h - crop_size))
    return int(xoff), int(yoff)


def crop_rect_mask(frame_h, frame_w, xoff, yoff, crop_size):
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    mask[yoff:yoff + crop_size, xoff:xoff + crop_size] = True
    return mask


# =========================================================
# Motion helpers
# =========================================================
def downscaled_mask_count(mask, downscale=4):
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    target_wh = (max(2, w // downscale), max(2, h // downscale))
    mask_ds = cv2.resize(mask, target_wh, interpolation=cv2.INTER_NEAREST).astype(bool)
    return int(mask_ds.sum())


def _prep_gray_for_registration(frame, downscale=4):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if downscale > 1:
        h, w = gray.shape
        gray = cv2.resize(
            gray,
            (max(2, w // downscale), max(2, h // downscale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = gray.astype(np.float32) / 255.0
    return gray


def _build_ecc_reference_from_cap(cap, downscale=4, n_ref_frames=30, max_frames=None):
    frames = []
    idx = 0

    while len(frames) < int(n_ref_frames):
        if max_frames is not None and idx >= int(max_frames):
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frames.append(_prep_gray_for_registration(frame, downscale=downscale))
        idx += 1

    if len(frames) == 0:
        raise RuntimeError("Could not build ECC reference; no frames read.")

    return np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)


def estimate_camera_motion_ecc(
    video_path,
    registration_mask,
    downscale=4,
    n_ref_frames=30,
    max_frames=None,
):
    """
    Estimate frame translation using ONLY stable outside pixels.

    Returns dataframe with:
      frame, time_sec, dx_px, dy_px, motion_mag_px, ecc_score, ecc_failed
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    h, w = frame0.shape[:2]

    registration_mask = registration_mask.astype(bool)
    if registration_mask.shape != (h, w):
        cap.release()
        raise ValueError(
            f"registration_mask shape {registration_mask.shape} != frame shape {(h, w)}"
        )

    # Downscale registration mask to match ECC images
    target_wh = (max(2, w // downscale), max(2, h // downscale))
    reg_mask_small = cv2.resize(
        registration_mask.astype(np.uint8),
        target_wh,
        interpolation=cv2.INTER_NEAREST,
    )

    if int(reg_mask_small.sum()) == 0:
        cap.release()
        raise RuntimeError("registration_mask is empty after downscaling.")

    # Build stable reference from early frames using the same open capture.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ref = _build_ecc_reference_from_cap(
        cap,
        downscale=downscale,
        n_ref_frames=n_ref_frames,
        max_frames=max_frames,
    )

    # Restart video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    warp_mode = cv2.MOTION_TRANSLATION
    prev_warp = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )

    records = []
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= int(max_frames):
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        moving = _prep_gray_for_registration(frame, downscale=downscale)
        failed = False

        try:
            cc, warp = cv2.findTransformECC(
                templateImage=ref,
                inputImage=moving,
                warpMatrix=prev_warp.copy(),
                motionType=warp_mode,
                criteria=criteria,
                inputMask=reg_mask_small,
                gaussFiltSize=5,
            )
            prev_warp = warp.copy()

        except cv2.error:
            # If ECC fails, reuse previous good transform
            warp = prev_warp.copy()
            cc = np.nan
            failed = True

        dx_px = float(warp[0, 2] * downscale)
        dy_px = float(warp[1, 2] * downscale)

        records.append({
            "frame": int(frame_idx),
            "time_sec": float(frame_idx / fps),
            "dx_px": dx_px,
            "dy_px": dy_px,
            "motion_mag_px": float(np.hypot(dx_px, dy_px)),
            "ecc_score": float(cc) if np.isfinite(cc) else np.nan,
            "ecc_failed": bool(failed),
        })

        frame_idx += 1

    cap.release()

    return pd.DataFrame(records)


def _correct_frame_translation(frame, dx, dy, correction_sign=-1):
    """
    Apply translation correction to one frame in memory.
    Usually correction_sign=-1 is correct.
    """
    h, w = frame.shape[:2]

    M = np.float32([
        [1, 0, correction_sign * dx],
        [0, 1, correction_sign * dy],
    ])

    return cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _camera_motion_row_for_frame(camera_motion_df, frame_idx):
    """
    Return the ECC transform row for a specific absolute video frame.
    estimate_camera_motion_ecc() creates one contiguous row per frame.
    """
    if camera_motion_df is None or len(camera_motion_df) == 0:
        raise RuntimeError("camera_motion_df is empty")

    frame_idx = int(frame_idx)

    if 0 <= frame_idx < len(camera_motion_df):
        row = camera_motion_df.iloc[frame_idx]
        row_frame = int(row.get("frame", frame_idx))
        if row_frame == frame_idx:
            return row

    if "frame" in camera_motion_df.columns:
        matched = camera_motion_df.loc[
            pd.to_numeric(
                camera_motion_df["frame"],
                errors="coerce",
            ).eq(frame_idx)
        ]
        if len(matched) > 0:
            return matched.iloc[0]

    raise IndexError(
        f"No camera-motion transform is available for frame {frame_idx}"
    )


def _edge_magnitude_from_gray_u8(gray_u8):
    """
    Edge-image kernel used by the edge-motion trace.
    Keep this numerically consistent with coarse_edge_motion_trace_masked().
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_u8)

    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    p99 = np.percentile(mag, 99)
    if np.isfinite(p99) and p99 > 0:
        mag = np.clip(mag / p99 * 255.0, 0, 255)

    return mag.astype(np.float32)


def compute_motion_traces_single_pass(
    videopath,
    mask_inside,
    mask_outside,
    mask_edge=None,
    downscale=4,
    max_frames=None,
):
    """
    Compute inside and outside motion traces in a single video pass.
    Optionally compute edge-control trace too when mask_edge is provided.

    This preserves the same operations as:
      - coarse_motion_trace_masked(..., mask_inside)
      - coarse_motion_trace_masked(..., mask_outside)
    - coarse_edge_motion_trace_masked(..., mask_edge) [optional]
    but avoids re-decoding the video three times.
    """
    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise RuntimeError("Could not read first frame")

    h0, w0 = frame0.shape[:2]

    mask_inside = np.asarray(mask_inside, dtype=bool)
    mask_outside = np.asarray(mask_outside, dtype=bool)
    use_edge = mask_edge is not None
    if use_edge:
        mask_edge = np.asarray(mask_edge, dtype=bool)

    mask_items = [
        ("mask_inside", mask_inside),
        ("mask_outside", mask_outside),
    ]
    if use_edge:
        mask_items.append(("mask_edge", mask_edge))

    for mask_name, mask_arr in mask_items:
        if mask_arr.shape != (h0, w0):
            cap.release()
            raise ValueError(
                f"{mask_name} shape {mask_arr.shape} != frame shape {(h0, w0)}"
            )

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    if downscale > 1:
        target_wh = (max(2, w0 // downscale), max(2, h0 // downscale))
        gray0_ds = cv2.resize(gray0, target_wh, interpolation=cv2.INTER_AREA)

        inside_ds = cv2.resize(
            mask_inside.astype(np.uint8),
            target_wh,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        outside_ds = cv2.resize(
            mask_outside.astype(np.uint8),
            target_wh,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if use_edge:
            edge_ds = cv2.resize(
                mask_edge.astype(np.uint8),
                target_wh,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            edge_ds = None
    else:
        gray0_ds = gray0
        inside_ds = mask_inside.copy()
        outside_ds = mask_outside.copy()
        edge_ds = mask_edge.copy() if use_edge else None

    if int(inside_ds.sum()) == 0:
        cap.release()
        raise RuntimeError("Inside mask has no positive pixels after resizing")
    if int(outside_ds.sum()) == 0:
        cap.release()
        raise RuntimeError("Outside mask has no positive pixels after resizing")
    if use_edge and int(edge_ds.sum()) == 0:
        cap.release()
        raise RuntimeError("Edge-control mask has no positive pixels after resizing")

    prev_gray_ds = gray0_ds

    prev_edge = _edge_magnitude_from_gray_u8(gray0_ds) if use_edge else None

    motion_inside = []
    motion_outside = []
    motion_edge = []
    frame_idxs = []

    idx = 1
    while True:
        if max_frames is not None and idx >= int(max_frames):
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            gray_ds = cv2.resize(gray, (prev_gray_ds.shape[1], prev_gray_ds.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            gray_ds = gray

        diff_gray = cv2.absdiff(gray_ds, prev_gray_ds)
        motion_inside.append(float(diff_gray[inside_ds].sum()))
        motion_outside.append(float(diff_gray[outside_ds].sum()))

        if use_edge:
            cur_edge = _edge_magnitude_from_gray_u8(gray_ds)
            diff_edge = np.abs(cur_edge - prev_edge)
            motion_edge.append(float(diff_edge[edge_ds].sum()))
            prev_edge = cur_edge

        frame_idxs.append(idx)
        prev_gray_ds = gray_ds
        idx += 1

    cap.release()

    frame_idxs = np.asarray(frame_idxs, dtype=np.int32)

    res_inside = {
        "fps": float(fps),
        "frame_idxs": frame_idxs,
        "time_sec": frame_idxs / float(fps),
        "motion": np.asarray(motion_inside, dtype=np.float64),
        "n_used_pairs": int(len(motion_inside)),
    }
    res_outside = {
        "fps": float(fps),
        "frame_idxs": frame_idxs,
        "time_sec": frame_idxs / float(fps),
        "motion": np.asarray(motion_outside, dtype=np.float64),
        "n_used_pairs": int(len(motion_outside)),
    }
    res_edge = None
    if use_edge:
        res_edge = {
            "fps": float(fps),
            "frame_idxs": frame_idxs,
            "time_sec": frame_idxs / float(fps),
            "motion": np.asarray(motion_edge, dtype=np.float64),
            "n_used_pairs": int(len(motion_edge)),
        }

    return res_inside, res_outside, res_edge


def make_plate_edge_control_mask(
    rgb,
    mask_outside_circle,
    mask_ring=None,
    dist2=None,
    R=None,
    R_outer=None,
    edge_percentile=85,
    min_edge_pixels=800,
):
    """
    Build a plate-motion control mask from strong edge pixels.

    Preferred:
      - strong edges in the far outside region only

    Fallback:
      - if the outside region has too few strong edges, add only the
        outermost part of the ring/buffer, not the full ring.

    This avoids fish/reflection contamination in the ring buffer.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Improve local contrast for bright/flat wells.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    outside_candidate = np.asarray(mask_outside_circle, dtype=bool)
    edge_mask = np.zeros_like(outside_candidate, dtype=bool)

    vals = mag[outside_candidate]
    vals = vals[np.isfinite(vals)]

    if vals.size > 0:
        thr = np.percentile(vals, edge_percentile)
        edge_mask = outside_candidate & (mag >= thr)

    used_fallback_outer_ring = False

    # Fallback: if outside has too few edge pixels, add only the OUTERMOST
    # 30% of the ring/buffer. This is safer than the full ring because fish
    # reflections are usually closer to the inner rim.
    if int(edge_mask.sum()) < int(min_edge_pixels) and mask_ring is not None:
        used_fallback_outer_ring = True

        if dist2 is not None and R is not None and R_outer is not None:
            R_safe = float(R) + 0.70 * (float(R_outer) - float(R))
            outer_ring_only = (
                (dist2 > R_safe**2)
                & (dist2 <= float(R_outer)**2)
            )
        else:
            outer_ring_only = np.asarray(mask_ring, dtype=bool)

        fallback_candidate = outside_candidate | outer_ring_only

        vals2 = mag[fallback_candidate]
        vals2 = vals2[np.isfinite(vals2)]

        if vals2.size > 0:
            thr2 = np.percentile(vals2, edge_percentile)
            edge_mask = fallback_candidate & (mag >= thr2)

    info = {
        "edge_percentile": float(edge_percentile),
        "min_edge_pixels": int(min_edge_pixels),
        "n_edge_pixels": int(edge_mask.sum()),
        "used_fallback_outer_ring": int(used_fallback_outer_ring),
    }

    return edge_mask.astype(bool), info


def save_mask_overlay(
    rgb,
    mask,
    out_path,
    title="Mask overlay",
    color=(255, 0, 255),
    alpha=0.35,
):
    """
    Save a simple QC overlay for a binary mask.
    """
    rgb = np.asarray(rgb).copy()
    overlay = rgb.astype(np.float32).copy()

    mask = np.asarray(mask, dtype=bool)
    color_arr = np.array(color, dtype=np.float32)

    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color_arr
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_inside_ring_outside_overlay(
    rgb,
    mask_inside,
    mask_ring,
    mask_outside,
    out_path,
    title="Inside / ring / outside overlay",
):
    """
    Save QC overlay showing:
      inside = green
      ring buffer = yellow/orange
      outside = blue
    """
    rgb = np.asarray(rgb).copy()
    overlay = rgb.astype(np.float32).copy()

    mask_inside = np.asarray(mask_inside, dtype=bool)
    mask_ring = np.asarray(mask_ring, dtype=bool)
    mask_outside = np.asarray(mask_outside, dtype=bool)

    inside_color = np.array([0, 255, 0], dtype=np.float32)
    ring_color = np.array([255, 190, 0], dtype=np.float32)
    outside_color = np.array([0, 120, 255], dtype=np.float32)

    overlay[mask_outside] = 0.82 * overlay[mask_outside] + 0.18 * outside_color
    overlay[mask_ring] = 0.68 * overlay[mask_ring] + 0.32 * ring_color
    overlay[mask_inside] = 0.68 * overlay[mask_inside] + 0.32 * inside_color

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    legend_handles = [
        plt.Line2D([0], [0], color=np.array([0, 255, 0]) / 255.0, lw=6, label="Inside circle"),
        plt.Line2D([0], [0], color=np.array([255, 190, 0]) / 255.0, lw=6, label="Ring buffer"),
        plt.Line2D([0], [0], color=np.array([0, 120, 255]) / 255.0, lw=6, label="Outside region"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def get_state_epochs(state, t, frame_idxs):
    state = np.asarray(state, dtype=int)
    t = np.asarray(t, dtype=float)
    frame_idxs = np.asarray(frame_idxs, dtype=int)

    if state.size == 0:
        return []

    epochs = []
    start = 0
    current = int(state[0])

    for i in range(1, len(state)):
        if int(state[i]) != current:
            epochs.append({
                "state": int(current),
                "start_idx": int(start),
                "end_idx": int(i - 1),
                "start_frame": int(frame_idxs[start]),
                "end_frame": int(frame_idxs[i - 1]),
                "start_time": float(t[start]),
                "end_time": float(t[i - 1]),
                "duration_sec": float(t[i - 1] - t[start]),
                "n_points": int(i - start),
            })
            start = i
            current = int(state[i])

    epochs.append({
        "state": int(current),
        "start_idx": int(start),
        "end_idx": int(len(state) - 1),
        "start_frame": int(frame_idxs[start]),
        "end_frame": int(frame_idxs[-1]),
        "start_time": float(t[start]),
        "end_time": float(t[-1]),
        "duration_sec": float(t[-1] - t[start]),
        "n_points": int(len(state) - start),
    })
    return epochs


# =========================================================
# Plot helpers
# =========================================================
def save_ring_debug_plot(rgb, gray, binary, comp_mask, ring_mask, res, out_path):
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Original")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(gray, cmap="gray")
    ax[0, 1].set_title("Gray")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(binary, cmap="gray")
    ax[0, 2].set_title(f"Binary (p_dark={res['p_dark']}, thr={res['thr']:.1f})")
    ax[0, 2].axis("off")

    ax[1, 0].imshow(comp_mask, cmap="gray")
    ax[1, 0].set_title("Selected Main Annular Component")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(ring_mask, cmap="gray")
    ax[1, 1].set_title("Thin Band Used for Fit")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(rgb)
    circle = plt.Circle((res["cx"], res["cy"]), res["radius"], color="cyan", fill=False, linewidth=2)
    ax[1, 2].add_patch(circle)
    ax[1, 2].plot(res["cx"], res["cy"], "bo", markersize=5)

    comp_u8 = comp_mask.astype(np.uint8)
    contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        xy = cnt[:, 0, :]
        ax[1, 2].plot(xy[:, 0], xy[:, 1], color="red", linewidth=1.0)

    ax[1, 2].set_title(f"Final Fit (RMSE={res['rmse']:.2f}, score={res['score']:.2f})")
    ax[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_inside_outside_diagnostic_plot(
    res_in, res_out, in_norm, out_norm, motion_sub, well, out_path, beta=None, alpha=None
):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax[0].plot(res_in["time_sec"], res_in["motion"], label="Inside circle", alpha=0.9)
    ax[0].plot(res_out["time_sec"], res_out["motion"], label="Outside circle", alpha=0.9)
    ax[0].set_ylabel("Raw motion\n(sum absdiff)")
    ax[0].set_title(f"{well}: inside vs outside coarse motion")
    ax[0].legend(frameon=False)

    ax[1].plot(res_in["time_sec"], in_norm, label="Inside / pixel", alpha=0.9)
    ax[1].plot(res_out["time_sec"], out_norm, label="Outside / pixel", alpha=0.9)

    if beta is not None and alpha is not None:
        out_fit = beta * out_norm + alpha
        ax[1].plot(
            res_in["time_sec"], out_fit,
            label=f"Fitted outside → inside (β={beta:.3f}, α={alpha:.3e})",
            alpha=0.9,
        )

    ax[1].set_ylabel("Normalized motion")
    ax[1].set_title("Normalized inside vs outside")
    ax[1].legend(frameon=False)

    ax[2].plot(res_in["time_sec"], motion_sub, label="Inside - fitted outside", linewidth=1.2)
    ax[2].axhline(0, linestyle="--", linewidth=1)
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Residual")
    ax[2].set_title("Motion residual: inside - fitted outside")
    ax[2].legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def state_name(state_val):
    if state_val == -1:
        return "BELOW_T2_LOW"
    elif state_val == 0:
        return "BASELINE"
    elif state_val == 1:
        return "ABOVE_T2_HIGH"
    return f"STATE_{state_val}"


# =========================================================
# Epoch-restricted frame readers
# =========================================================
def mean_trace_from_blob_epoch(
    videopath: str,
    xoff: int,
    yoff: int,
    crop_size: int,
    blob_mask: np.ndarray,
    frame_start: int,
    frame_end: int,
    camera_motion_df=None,
    correction_sign: int = -1,
    epoch_gray_cube: np.ndarray = None,
    epoch_frames: np.ndarray = None,
):
    """
    Extract a blob intensity trace from the requested epoch.

    When camera_motion_df is supplied, each frame is translation-corrected
    in memory before cropping and trace extraction. No corrected video is
    saved.
    """
    blob_mask = np.asarray(blob_mask, dtype=bool)
    ys, xs = np.where(blob_mask)

    if ys.size == 0:
        raise RuntimeError("blob_mask is empty")

    if epoch_gray_cube is not None:
        cube = np.asarray(epoch_gray_cube)
        if cube.ndim != 3:
            raise RuntimeError(
                f"epoch_gray_cube must be 3D (T,H,W); got {cube.shape}"
            )

        if cube.shape[1] != crop_size or cube.shape[2] != crop_size:
            raise RuntimeError(
                f"epoch_gray_cube spatial shape {cube.shape[1:]} does not match "
                f"crop_size {(crop_size, crop_size)}"
            )

        if epoch_frames is None:
            frames = np.arange(
                int(frame_start),
                int(frame_end) + 1,
                dtype=np.int32,
            )
        else:
            frames = np.asarray(epoch_frames, dtype=np.int32)

        if cube.shape[0] != len(frames):
            raise RuntimeError(
                f"epoch_gray_cube length {cube.shape[0]} != epoch_frames "
                f"length {len(frames)}"
            )

        trace = cube[:, ys, xs].mean(axis=1).astype(np.float64)
        return frames, trace

    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    frames = []
    trace = []
    f = int(frame_start)

    while f <= int(frame_end):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        if camera_motion_df is not None:
            motion_row = _camera_motion_row_for_frame(
                camera_motion_df,
                f,
            )
            frame_bgr = _correct_frame_translation(
                frame_bgr,
                dx=float(motion_row["dx_px"]),
                dy=float(motion_row["dy_px"]),
                correction_sign=int(correction_sign),
            )

        crop = frame_bgr[
            yoff:yoff + crop_size,
            xoff:xoff + crop_size,
        ]
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            cap.release()
            raise RuntimeError(
                f"Crop shape mismatch at frame {f}: got {crop.shape[:2]}, "
                f"expected {(crop_size, crop_size)}"
            )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        trace.append(float(gray[ys, xs].mean()))
        frames.append(f)
        f += 1

    cap.release()

    if len(trace) == 0:
        raise RuntimeError("No frames read for blob trace")

    return (
        np.asarray(frames, dtype=np.int32),
        np.asarray(trace, dtype=np.float64),
    )


def build_X_from_crop_mask_epoch(
    videopath: str,
    mask_crop: np.ndarray,
    xoff: int,
    yoff: int,
    frame_start: int,
    frame_end: int,
    camera_motion_df=None,
    correction_sign: int = -1,
    epoch_gray_cube: np.ndarray = None,
):
    """
    Build the epoch time-by-pixel matrix.

    When camera_motion_df is supplied, each full frame is corrected in
    memory before the fixed crop and fish-mask pixels are extracted.
    No corrected video is written.
    """
    mask = mask_crop.astype(bool)
    ys, xs = np.where(mask)

    if ys.size == 0:
        raise RuntimeError("mask_crop has no True pixels")

    H, W = mask.shape

    if epoch_gray_cube is not None:
        cube = np.asarray(epoch_gray_cube)
        if cube.ndim != 3:
            raise RuntimeError(
                f"epoch_gray_cube must be 3D (T,H,W); got {cube.shape}"
            )
        if cube.shape[1] != H or cube.shape[2] != W:
            raise RuntimeError(
                f"epoch_gray_cube spatial shape {cube.shape[1:]} does not "
                f"match mask shape {(H, W)}"
            )

        X = cube[:, ys, xs].astype(np.float64)
        return X, ys, xs

    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {videopath}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    X_rows = []
    f = int(frame_start)

    while f <= int(frame_end):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        if camera_motion_df is not None:
            motion_row = _camera_motion_row_for_frame(
                camera_motion_df,
                f,
            )
            frame_bgr = _correct_frame_translation(
                frame_bgr,
                dx=float(motion_row["dx_px"]),
                dy=float(motion_row["dy_px"]),
                correction_sign=int(correction_sign),
            )

        crop = frame_bgr[
            yoff:yoff + H,
            xoff:xoff + W,
        ]
        if crop.shape[0] != H or crop.shape[1] != W:
            cap.release()
            raise RuntimeError(
                f"Crop shape mismatch while building X: "
                f"expected {(H, W)}, got {crop.shape[:2]}"
            )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)

        X_rows.append(gray[ys, xs])
        f += 1

    cap.release()

    if len(X_rows) == 0:
        raise RuntimeError("No frames read while building X")

    X = np.stack(X_rows, axis=0).astype(np.float64)
    return X, ys, xs


def load_epoch_gray_cube(
    videopath: str,
    xoff: int,
    yoff: int,
    crop_size: int,
    frame_start: int,
    frame_end: int,
    camera_motion_df=None,
    correction_sign: int = -1,
):
    """
    Load epoch frames once as crop-space grayscale cube for fallback reuse.

    When camera_motion_df is supplied, correction is applied in memory before
    cropping so downstream ICA/blob traces share exactly the same corrected data.
    """
    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))

    frames = []
    cube_rows = []

    f = int(frame_start)
    while f <= int(frame_end):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        if camera_motion_df is not None:
            motion_row = _camera_motion_row_for_frame(
                camera_motion_df,
                f,
            )
            frame_bgr = _correct_frame_translation(
                frame_bgr,
                dx=float(motion_row["dx_px"]),
                dy=float(motion_row["dy_px"]),
                correction_sign=int(correction_sign),
            )

        crop = frame_bgr[
            yoff:yoff + crop_size,
            xoff:xoff + crop_size,
        ]
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            cap.release()
            raise RuntimeError(
                f"Crop shape mismatch while loading epoch cube: "
                f"expected {(crop_size, crop_size)}, got {crop.shape[:2]}"
            )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)

        frames.append(f)
        cube_rows.append(gray)
        f += 1

    cap.release()

    if len(cube_rows) == 0:
        raise RuntimeError("No frames read while loading epoch cube")

    return (
        np.asarray(frames, dtype=np.int32),
        np.stack(cube_rows, axis=0).astype(np.uint8),
    )



def center_scale(Xin: np.ndarray) -> np.ndarray:
    Xout = Xin.astype(np.float64, copy=True)
    Xout -= Xout.mean(axis=0, keepdims=True)
    s = Xout.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    Xout /= s
    return Xout

# =========================================================
# Blob / ICA helpers
# =========================================================
def top_blobs_from_smap(
    smap,
    mask_crop,
    percentile=90,
    use_abs=True,
    min_pixels=110,
    fill_holes=True,
    top_k=5,
):
    valid = mask_crop.astype(bool) & np.isfinite(smap)

    blobs = []

    for sign_name, sign_mask, work in [
        ("positive", smap > 0, smap),
        ("negative", smap < 0, -smap),
    ]:
        valid_sign = valid & sign_mask
        vals = work[valid_sign]

        if vals.size == 0:
            continue

        thresh = np.percentile(vals, percentile)

        bw = np.zeros_like(smap, dtype=bool)
        bw[valid_sign] = work[valid_sign] >= thresh

        lbl, nlab = ndimage.label(bw)

        for lab in range(1, nlab + 1):
            blob_mask = lbl == lab
            raw_size = int(blob_mask.sum())

            if raw_size < min_pixels:
                continue

            blob_filled = (
                ndimage.binary_fill_holes(blob_mask)
                if fill_holes else blob_mask.copy()
            )

            blob_ys, blob_xs = np.where(blob_filled)
            if len(blob_xs) == 0:
                continue

            linearity_pca = np.nan
            if len(blob_xs) >= 2:
                pts = np.column_stack([blob_xs, blob_ys]).astype(float)
                center = pts.mean(axis=0)
                pts0 = pts - center
                cov = np.cov(pts0, rowvar=False)
                evals, _ = np.linalg.eigh(cov)
                evals = np.sort(evals)[::-1]
                if evals[0] > 0:
                    linearity_pca = float(1.0 - (evals[1] / evals[0]))

            blobs.append({
                "rank": 0,
                "label": int(lab),
                "blob_sign": sign_name,
                "blob_mask": blob_mask,
                "blob_filled": blob_filled,
                "blob_size": raw_size,
                "linearity_pca": linearity_pca,
            })

    # if len(blobs) == 0:
    #     raise RuntimeError(f"No signed blobs >= {min_pixels} pixels found")

    # blobs = sorted(blobs, key=lambda b: b["blob_size"], reverse=True)[:top_k]
    if len(blobs) == 0:
        raise RuntimeError(
            f"No signed blobs >= {min_pixels} pixels found"
        )

    # Keep only blobs that satisfy the shape requirement
    eligible_blobs = [
        b for b in blobs
        if (
            np.isfinite(b["linearity_pca"])
            and b["linearity_pca"] <= 0.95
        )
    ]

    if len(eligible_blobs) == 0:
        raise RuntimeError(
            f"No signed blobs >= {min_pixels} pixels "
            f"with linearity_pca <= 0.90 found"
        )

    # Select the five largest eligible blobs
    blobs = sorted(
        eligible_blobs,
        key=lambda b: b["blob_size"],
        reverse=True,
    )[:top_k]

    for i, b in enumerate(blobs, start=1):
        b["rank"] = i

    return {
        "smap": smap,
        "work": np.abs(smap),
        "threshold": np.nan,
        "binary_mask": np.logical_or.reduce([b["blob_mask"] for b in blobs]),
        "n_labels": len(blobs),
        "all_sizes": np.array([b["blob_size"] for b in blobs]),
        "blobs": blobs,
    }


def reconstruct_and_analyze_blob_signal_epoch(
    blob_mask,
    videopath,
    xoff,
    yoff,
    crop_size,
    fps_use,
    frame_start,
    frame_end,
    low=0.7,
    high=10.0,
    order=3,
    acf_periodicity_thresh=0.50,
    min_lag_sec=0.3,
    max_lag_sec=2.0,
    prom_norm_thresh=0.40,
    min_distance_sec=0.3,
    camera_motion_df=None,
    correction_sign=-1,
    epoch_gray_cube: np.ndarray = None,
    epoch_frames: np.ndarray = None,
):
    min_lag = max(1, int(round(min_lag_sec * fps_use)))
    dt = 1.0 / fps_use
    distance = max(1, int(round(min_distance_sec * fps_use)))

    b, a = signal.butter(order, [low, high], btype="bandpass", fs=fps_use)

    frames_blob, trace_blob = mean_trace_from_blob_epoch(
        videopath=videopath,
        xoff=xoff,
        yoff=yoff,
        crop_size=crop_size,
        blob_mask=blob_mask,
        frame_start=frame_start,
        frame_end=frame_end,
        camera_motion_df=camera_motion_df,
        correction_sign=correction_sign,
        epoch_gray_cube=epoch_gray_cube,
        epoch_frames=epoch_frames,
    )

    if len(trace_blob) < 10:
        raise RuntimeError("trace too short")

    trace_raw = trace_blob.copy()
    trace_centered = trace_blob - np.mean(trace_blob)
    trace_filt = signal.filtfilt(b, a, trace_centered)

    analytic = signal.hilbert(trace_filt)
    trace_hilbert = np.abs(analytic)

    max_amp_hilbert = float(np.max(trace_hilbert))
    integral_abs = float(np.trapezoid(np.abs(trace_filt), dx=dt))
    rms = float(np.sqrt(np.mean(trace_filt ** 2)))

    trace_std = np.std(trace_filt)
    if not np.isfinite(trace_std) or trace_std <= 0:
        raise RuntimeError("trace_std_invalid")

    trace_norm = (trace_filt - np.mean(trace_filt)) / (trace_std + 1e-8)

    acf_full = signal.correlate(trace_norm, trace_norm, mode="full", method="fft")
    acf = acf_full[len(acf_full) // 2:]
    acf = acf / (acf[0] + 1e-8)

    max_lag = min(int(round(max_lag_sec * fps_use)), len(acf) - 1)
    if max_lag <= min_lag:
        raise RuntimeError("invalid lag window for ACF")

    acf_window = acf[min_lag:max_lag + 1]
    acf_peaks, _ = signal.find_peaks(acf_window)

    periodicity_score = 0.0
    best_lag = None
    best_freq_hz = np.nan
    best_bpm = np.nan

    if len(acf_peaks) > 0:
        peak_vals = acf_window[acf_peaks]
        best_idx = np.argmax(peak_vals)
        best_peak_local = acf_peaks[best_idx]
        periodicity_score = float(peak_vals[best_idx])
        best_lag = int(min_lag + best_peak_local)
        best_period_sec = float(best_lag / fps_use)
        best_freq_hz = float(1.0 / best_period_sec) if best_period_sec > 0 else np.nan
        best_bpm = float(best_freq_hz * 60.0) if np.isfinite(best_freq_hz) else np.nan

    is_periodic = bool(periodicity_score >= acf_periodicity_thresh)

    if is_periodic:
        peak_signal = trace_filt
        peak_signal_name = "bandpassed"
    else:
        peak_signal = trace_hilbert
        peak_signal_name = "hilbert_envelope"

    peaks, _ = signal.find_peaks(peak_signal, distance=distance)
    real_peaks = np.array([], dtype=int)

    if len(peaks) > 0:
        prominences = signal.peak_prominences(peak_signal, peaks)[0]
        p99 = np.percentile(peak_signal, 99)
        if np.isfinite(p99) and p99 > 0:
            prominence_norm_p99 = prominences / (p99 + 1e-8)
            real_mask = prominence_norm_p99 > prom_norm_thresh
            real_peaks = peaks[real_mask]

    interval_cv = np.nan
    if len(real_peaks) >= 3:
        rr_sec = np.diff(real_peaks) / fps_use
        interval_cv = float(np.std(rr_sec) / (np.mean(rr_sec) + 1e-8))

    return {
        "frames_blob": frames_blob,
        "trace_raw": trace_raw,
        "trace_filt": trace_filt,
        "trace_hilbert": trace_hilbert,
        "max_amp_hilbert": max_amp_hilbert,
        "integral_abs": integral_abs,
        "rms": rms,
        "acf": acf,
        "lags_sec": np.arange(len(acf), dtype=np.float64) / fps_use,
        "periodicity_score": periodicity_score,
        "is_periodic": is_periodic,
        "best_lag": best_lag,
        "best_freq_hz": best_freq_hz,
        "best_bpm": best_bpm,
        "peak_signal_name": peak_signal_name,
        "n_peaks_all": int(len(peaks)),
        "n_peaks_real": int(len(real_peaks)),
        "interval_cv": interval_cv,
    }



# =========================================================
# Fixed-length 235-feature builder for fallback XGBoost
# =========================================================
FIXED_FEATURE_N_ICS = 5
FIXED_FEATURE_N_BLOBS_PER_IC = 5
FIXED_BLOB_FEATURES = [
    "periodicity_score",
    "n_peaks_real",
    "n_peaks_all",
    "interval_cv",
    "rms",
    "integral_abs",
    "max_amp_hilbert",
    "blob_size",
    "linearity_pca",
]


def fixed_235_feature_names():
    """
    Return the exact predictor order used by build_fixed_length_fish_vectors.py.

    Layout:
      5 ranked ICs x [energy + n_blobs + 5 ranked blobs x 9 blob features]
      = 235 predictors.
    """
    feature_cols = []
    for ic_rank in range(1, FIXED_FEATURE_N_ICS + 1):
        prefix = f"rankIC{ic_rank}"
        feature_cols.extend([
            f"{prefix}_energy",
            f"{prefix}_n_blobs",
        ])
        for blob_rank in range(1, FIXED_FEATURE_N_BLOBS_PER_IC + 1):
            feature_cols.extend([
                f"{prefix}_blob{blob_rank}_{feat}"
                for feat in FIXED_BLOB_FEATURES
            ])

    if len(feature_cols) != 235:
        raise RuntimeError(
            f"Internal feature-layout error: expected 235 predictors, "
            f"got {len(feature_cols)}"
        )
    return feature_cols


def build_fixed_235_feature_vector_from_fallback(
    all_blob_rows,
    normalized_energy_rows,
):
    """
    Build one fish-level 235-feature vector directly from fallback outputs.

    This intentionally mirrors build_fixed_length_fish_vectors.py:
      - IC universe = union of ICs in blob rows and normalized-energy rows
      - IC rank = original_normalized_energy descending, original IC ascending
      - retain top 5 ICs
      - ic_n_blobs = number of blob-summary rows for that IC
      - blob rank within IC = blob_size descending (stable mergesort)
      - retain top 5 blobs/IC
      - missing IC/blob slots are padded with NaN
      - missing IC slots use n_blobs = 0
    """
    feature_cols = fixed_235_feature_names()

    blob_df = pd.DataFrame(list(all_blob_rows)).copy()
    energy_df = pd.DataFrame(list(normalized_energy_rows)).copy()

    # Normalize required columns exactly as the standalone builder does.
    if "ic" not in blob_df.columns:
        blob_df["ic"] = np.nan
    blob_df["ic"] = pd.to_numeric(blob_df["ic"], errors="coerce")

    for col in FIXED_BLOB_FEATURES:
        if col not in blob_df.columns:
            blob_df[col] = np.nan
        blob_df[col] = pd.to_numeric(blob_df[col], errors="coerce")

    if "ic" not in energy_df.columns:
        energy_df["ic"] = np.nan
    if "original_normalized_energy" not in energy_df.columns:
        energy_df["original_normalized_energy"] = np.nan

    energy_df["ic"] = pd.to_numeric(energy_df["ic"], errors="coerce")
    energy_df["original_normalized_energy"] = pd.to_numeric(
        energy_df["original_normalized_energy"],
        errors="coerce",
    )

    # Standalone builder keeps the last duplicate energy row for an IC.
    energy_df = energy_df.drop_duplicates(subset=["ic"], keep="last")

    # IC universe = union(blob ICs, energy ICs).
    blob_ics = (
        blob_df[["ic"]].drop_duplicates()
        if len(blob_df) > 0
        else pd.DataFrame(columns=["ic"])
    )
    energy_ics = (
        energy_df[["ic"]].drop_duplicates()
        if len(energy_df) > 0
        else pd.DataFrame(columns=["ic"])
    )
    ic_table = pd.concat(
        [blob_ics, energy_ics],
        ignore_index=True,
    ).drop_duplicates()

    if len(ic_table) > 0:
        blob_counts = (
            blob_df.groupby("ic", dropna=False)
            .size()
            .reset_index(name="ic_n_blobs")
        )

        ic_table = ic_table.merge(
            blob_counts,
            on="ic",
            how="left",
        )
        ic_table["ic_n_blobs"] = (
            pd.to_numeric(ic_table["ic_n_blobs"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        ic_table = ic_table.merge(
            energy_df[["ic", "original_normalized_energy"]],
            on="ic",
            how="left",
            validate="one_to_one",
        )

        # Rank ICs exactly as standalone builder:
        # energy descending, IC ascending, NaNs last, stable sort.
        ic_table["_energy_sort"] = pd.to_numeric(
            ic_table["original_normalized_energy"],
            errors="coerce",
        )
        ic_table["_ic_sort"] = pd.to_numeric(
            ic_table["ic"],
            errors="coerce",
        )
        ic_table = (
            ic_table.sort_values(
                by=["_energy_sort", "_ic_sort"],
                ascending=[False, True],
                na_position="last",
                kind="mergesort",
            )
            .drop(columns=["_energy_sort", "_ic_sort"])
            .reset_index(drop=True)
        )
    else:
        ic_table = pd.DataFrame(
            columns=[
                "ic",
                "ic_n_blobs",
                "original_normalized_energy",
            ]
        )

    selected_ics = ic_table.head(FIXED_FEATURE_N_ICS)

    row = {}

    for ic_rank in range(1, FIXED_FEATURE_N_ICS + 1):
        prefix = f"rankIC{ic_rank}"

        # Exact standalone padding.
        row[f"{prefix}_energy"] = np.nan
        row[f"{prefix}_n_blobs"] = 0

        for blob_rank in range(1, FIXED_FEATURE_N_BLOBS_PER_IC + 1):
            for feat in FIXED_BLOB_FEATURES:
                row[f"{prefix}_blob{blob_rank}_{feat}"] = np.nan

        if ic_rank > len(selected_ics):
            continue

        ic_info = selected_ics.iloc[ic_rank - 1]
        original_ic = ic_info["ic"]

        row[f"{prefix}_energy"] = ic_info["original_normalized_energy"]
        row[f"{prefix}_n_blobs"] = int(ic_info["ic_n_blobs"])

        # Pull blobs for this original IC.
        if pd.isna(original_ic):
            g = blob_df[blob_df["ic"].isna()].copy()
        else:
            g = blob_df[blob_df["ic"].eq(original_ic)].copy()

        if "blob_size" not in g.columns:
            g["blob_size"] = np.nan
        g["blob_size"] = pd.to_numeric(g["blob_size"], errors="coerce")

        # Exact standalone blob ranking: blob_size descending, NaNs last.
        g = (
            g.sort_values(
                by="blob_size",
                ascending=False,
                na_position="last",
                kind="mergesort",
            )
            .head(FIXED_FEATURE_N_BLOBS_PER_IC)
            .reset_index(drop=True)
        )

        for blob_rank in range(1, FIXED_FEATURE_N_BLOBS_PER_IC + 1):
            if blob_rank > len(g):
                continue

            blob = g.iloc[blob_rank - 1]
            for feat in FIXED_BLOB_FEATURES:
                value = blob[feat] if feat in blob.index else np.nan
                row[f"{prefix}_blob{blob_rank}_{feat}"] = (
                    value if pd.notna(value) else np.nan
                )

    feature_df = pd.DataFrame([row], columns=feature_cols)

    if feature_df.shape != (1, 235):
        raise RuntimeError(
            f"Expected fallback fixed feature shape (1, 235), "
            f"got {feature_df.shape}"
        )

    return feature_df, feature_cols




def prepare_xgboost_no_interval_cv_features(fixed_235_df):
    """
    Convert the raw 235-feature fallback vector into the exact predictor
    representation used by optimized_grouped_ml_collapsed_NO_INTERVAL_CV.

    Training-notebook contract:
      1) Start from the fixed fish-level feature table.
      2) Add:
           max_energy
           min_energy
           mean_energy
           has_periodic_blob
      3) Exclude every *_n_blobs predictor.
      4) Exclude every *_interval_cv predictor.
      5) Preserve deterministic column order.

    Expected predictor count:
        235 - 5 n_blobs - 25 interval_cv + 4 engineered = 209.
    """
    x = fixed_235_df.copy()

    raw_feature_cols = fixed_235_feature_names()
    missing_raw = [c for c in raw_feature_cols if c not in x.columns]
    if missing_raw:
        raise RuntimeError(
            "Fixed 235 table is missing required predictors: "
            + ", ".join(missing_raw[:20])
        )

    # Exact energy columns used in the training notebook.
    energy_cols = [
        "rankIC1_energy",
        "rankIC2_energy",
        "rankIC3_energy",
        "rankIC4_energy",
        "rankIC5_energy",
    ]
    x[energy_cols] = x[energy_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Fish-level IC-energy summary features.
    x["max_energy"] = x[energy_cols].max(axis=1)
    x["min_energy"] = x[energy_cols].min(axis=1)
    x["mean_energy"] = x[energy_cols].mean(axis=1)

    # Exact periodicity-derived feature from the training notebook.
    periodicity_cols = [
        c for c in x.columns
        if c.endswith("_periodicity_score")
    ]
    x["has_periodic_blob"] = (
        x[periodicity_cols]
        .ge(0.50)
        .any(axis=1)
        .astype(int)
    )

    # Match notebook feature filtering:
    #   - numeric fish-level predictors
    #   - exclude *_n_blobs
    #   - exclude *_interval_cv
    #
    # Starting from the known fixed feature layout avoids accidentally
    # including metadata or output/classification columns.
    model_feature_cols = [
        c for c in raw_feature_cols
        if not c.endswith("_n_blobs")
        and not c.endswith("_interval_cv")
    ]

    # Engineered features were appended to labeled_df before feature selection,
    # so they appear after the original fixed-vector predictors.
    model_feature_cols.extend([
        "max_energy",
        "min_energy",
        "mean_energy",
        "has_periodic_blob",
    ])

    model_df = x[model_feature_cols].copy()

    if model_df.shape[1] != 209:
        raise RuntimeError(
            f"Expected 209 XGBoost predictors for the no-interval_cv model, "
            f"got {model_df.shape[1]}"
        )

    if any(c.endswith("_n_blobs") for c in model_feature_cols):
        raise RuntimeError("Unexpected *_n_blobs predictor in XGBoost table.")

    if any(c.endswith("_interval_cv") for c in model_feature_cols):
        raise RuntimeError("Unexpected *_interval_cv predictor in XGBoost table.")

    return model_df, model_feature_cols




def predict_fallback_xgboost(
    X,
    model_path,
    mode="binary",
    alive_threshold=0.50,
):
    """
    Run the trained fallback XGBoost classifier on one 209-feature fish row.

    Modes
    -----
    binary:
        Training target:
            0 = Dead
            1 = Alive
        Final prediction:
            Alive when P(Alive) >= alive_threshold.

    collapsed:
        Training target:
            0 = Dead
            1 = Weak
            2 = Strong
        This exactly follows the training notebook:
            multiclass class = argmax(P_dead, P_weak, P_strong)
            collapsed Alive = predicted class is Weak or Strong
        P(Alive) is also reported as P(Weak) + P(Strong), but is not used
        as the class threshold for the collapsed strategy.
    """
    model_path = str(model_path).strip()
    if not model_path:
        raise RuntimeError(
            "XGBoost fallback was reached but --xgb_model_path is empty."
        )

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"XGBoost model file does not exist: {model_file}"
        )

    mode = str(mode).strip().lower()
    if mode not in {"binary", "collapsed"}:
        raise ValueError(
            f"Unknown fallback model mode: {mode}. "
            f"Expected 'binary' or 'collapsed'."
        )

    # Lazy import so motion-positive wells do not require XGBoost import/load.
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(str(model_file))

    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is not None and int(expected_n) != int(X.shape[1]):
        raise RuntimeError(
            f"XGBoost model expects {int(expected_n)} predictors, "
            f"but fallback produced {int(X.shape[1])}."
        )

    probs = np.asarray(model.predict_proba(X), dtype=float)
    if probs.ndim != 2 or probs.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected predict_proba shape: {probs.shape}"
        )

    if mode == "binary":
        if probs.shape[1] != 2:
            raise RuntimeError(
                f"Binary fallback model returned {probs.shape[1]} classes; "
                f"expected 2."
            )

        p_dead = float(probs[0, 0])
        p_alive = float(probs[0, 1])
        pred_binary = int(p_alive >= float(alive_threshold))

        return {
            "model_mode": "binary",
            "predicted_multiclass_int": np.nan,
            "predicted_multiclass_label": "",
            "predicted_binary": int(pred_binary),
            "predicted_binary_label": "Alive" if pred_binary else "Dead",
            "p_dead": p_dead,
            "p_alive": p_alive,
            "p_weak": np.nan,
            "p_strong": np.nan,
            "alive_threshold": float(alive_threshold),
        }

    # collapsed 3-class -> binary
    if probs.shape[1] != 3:
        raise RuntimeError(
            f"Collapsed fallback model returned {probs.shape[1]} classes; "
            f"expected 3."
        )

    p_dead = float(probs[0, 0])
    p_weak = float(probs[0, 1])
    p_strong = float(probs[0, 2])
    p_alive = float(p_weak + p_strong)

    # Exact notebook behavior:
    # pred_multi = argmax(probabilities)
    # pred_binary = pred_multi != dead
    pred_multi = int(np.argmax(probs[0]))
    class_labels = {
        0: "Dead",
        1: "Weak",
        2: "Strong",
    }
    pred_binary = int(pred_multi != 0)

    return {
        "model_mode": "collapsed",
        "predicted_multiclass_int": int(pred_multi),
        "predicted_multiclass_label": class_labels[pred_multi],
        "predicted_binary": int(pred_binary),
        "predicted_binary_label": "Alive" if pred_binary else "Dead",
        "p_dead": p_dead,
        "p_alive": p_alive,
        "p_weak": p_weak,
        "p_strong": p_strong,
        "alive_threshold": np.nan,
    }



def run_fallback_ica_on_crop_mask_epoch(
    videopath: str,
    mask_crop: np.ndarray,
    xoff: int,
    yoff: int,
    fps_use: float,
    frame_start: int,
    frame_end: int,
    n_keep: int,
    out_dir: Path,
    well: str,
    ica_fun: str = "exp",
    ica_max_iter: int = 1000,
    ica_tol: float = 1e-4,
    ica_random_state: int = 32,
    camera_motion_df=None,
    correction_sign: int = -1,
    xgb_model_path: str = DEFAULT_XGB_MODEL_PATH,
    fallback_model_mode: str = DEFAULT_FALLBACK_MODEL_MODE,
    xgb_alive_threshold: float = DEFAULT_XGB_ALIVE_THRESHOLD,
):
    low_fc = 0.7
    high_fc = 10.0

    mask = mask_crop.astype(bool)
    H, W = mask.shape
    crop_size = H

    # Cache corrected epoch frames up to a fixed duration budget.
    # This avoids repeated video reads in fallback for typical epochs.
    MAX_EPOCH_CACHE_SECONDS = 31.0
    n_epoch_frames_est = max(0, int(frame_end) - int(frame_start) + 1)
    max_cache_frames = max(1, int(round(MAX_EPOCH_CACHE_SECONDS * float(fps_use))))
    est_cache_bytes = int(n_epoch_frames_est) * int(H) * int(W) * 4

    epoch_frames = None
    epoch_gray_cube = None

    if n_epoch_frames_est <= max_cache_frames:
        epoch_frames, epoch_gray_cube = load_epoch_gray_cube(
            videopath=videopath,
            xoff=xoff,
            yoff=yoff,
            crop_size=crop_size,
            frame_start=frame_start,
            frame_end=frame_end,
            camera_motion_df=camera_motion_df,
            correction_sign=correction_sign,
        )
        print(
            f"[INFO] well {well}: using fallback epoch cache "
            f"({n_epoch_frames_est} frames, "
            f"~{est_cache_bytes / (1024**2):.1f} MB estimated)"
        )
    else:
        print(
            f"[INFO] well {well}: skipping fallback epoch cache "
            f"({n_epoch_frames_est} frames > {max_cache_frames} frame "
            f"limit for {MAX_EPOCH_CACHE_SECONDS:.1f}s)"
        )

    X, ys, xs = build_X_from_crop_mask_epoch(
        videopath=videopath,
        mask_crop=mask,
        xoff=xoff,
        yoff=yoff,
        frame_start=frame_start,
        frame_end=frame_end,
        camera_motion_df=camera_motion_df,
        correction_sign=correction_sign,
        epoch_gray_cube=epoch_gray_cube,
    )

    motion_correction_used = int(camera_motion_df is not None)

    print(
        f"[INFO] Epoch ICA well {well}: "
        f"motion_correction_used={motion_correction_used}, "
        f"correction_sign={int(correction_sign)}"
    )

    t_frames, p_pixels = X.shape
    print(f"[INFO] Epoch ICA well {well}: fish pixels={p_pixels}, X shape={X.shape}")

    b, a = signal.butter(2, [low_fc, high_fc], btype="bandpass", fs=fps_use)

    padlen = min(int(3 * fps_use / low_fc), X.shape[0] - 2)
    if padlen < 1:
        raise RuntimeError(f"padlen became invalid: {padlen}")

    X_pad = np.pad(X, ((padlen, padlen), (0, 0)), mode="reflect")
    X_bp = signal.filtfilt(b, a, X_pad, axis=0, padtype=None)[padlen:-padlen].copy()

    cube_bp = np.zeros((t_frames, H, W), dtype=np.float32)
    cube_bp[:, ys, xs] = X_bp.astype(np.float32)
    cube_g = median_filter(cube_bp, size=(1, 5, 5)).astype(np.float32)

    X_g_1 = cube_g[:, ys, xs].astype(np.float64)
    X_g_1_cs = center_scale(X_g_1)

    n_keep_eff = int(min(n_keep, X_g_1_cs.shape[0], X_g_1_cs.shape[1]))
    if n_keep_eff < 1:
        raise RuntimeError("n_keep_eff < 1")

    pca = PCA(n_components=n_keep_eff, random_state=ica_random_state)
    Z = pca.fit_transform(X_g_1_cs)


    ica = FastICA(
        n_components=n_keep_eff,
        fun=ica_fun,
        max_iter=ica_max_iter,
        tol=ica_tol,
        random_state=ica_random_state,
    )

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        S = ica.fit_transform(Z)

    ica_convergence_warning = any(
        issubclass(w.category, ConvergenceWarning) for w in wlist
    )
    ica_converged = not ica_convergence_warning
    ica_n_iter = getattr(ica, "n_iter_", np.nan)

    print(
        f"[ICA] well={well} converged={ica_converged} "
        f"n_iter={ica_n_iter} max_iter={ica_max_iter}"
    )

    A_z = ica.mixing_
    V = pca.components_.T
    A_pix = V @ A_z

    def spatial_map(ic_idx: int) -> np.ndarray:
        smap = np.zeros((H, W), dtype=np.float32)
        smap[ys, xs] = A_pix[:, ic_idx].astype(np.float32)
        return smap

    ica_root = out_dir / f"well_{well}_longest_baseline_epoch_fallback_ica"
    time_dir = ica_root / "time_signals"
    spatial_dir = ica_root / "spatial_maps"
    blob_dir = ica_root / "blob_summary"
    trace_dir = ica_root / "blob_traces"

    ensure_dir(ica_root)
    ensure_dir(time_dir)
    ensure_dir(spatial_dir)
    ensure_dir(blob_dir)
    ensure_dir(trace_dir)

    # -------------------------
    # ICA time signals
    # -------------------------
    time_pngs = []
    time_axis = (np.arange(S.shape[0], dtype=np.float64) + frame_start) / fps_use

    fig, axes = plt.subplots(n_keep_eff, 1, figsize=(10, 2.5 * n_keep_eff), sharex=True)
    axes = np.atleast_1d(axes)

    for ic in range(n_keep_eff):
        ax = axes[ic]
        s0 = S[:, ic] - np.mean(S[:, ic])
        ax.plot(time_axis, s0, linewidth=1.0)
        ax.set_ylabel(f"IC{ic}")
        ax.grid(alpha=0.2)
        ax.set_title(f"Well {well} — longest baseline epoch ICA signal IC{ic}")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    png = time_dir / f"well_{well}_epoch_ica_timesignals_all.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    time_pngs.append(str(png))


    # -------------------------
    # ICA spatial maps
    # -------------------------
    spatial_pngs = []

    fig, axes = plt.subplots(1, n_keep_eff, figsize=(4.5 * n_keep_eff, 4.5))
    axes = np.atleast_1d(axes)

    for ic in range(n_keep_eff):
        ax = axes[ic]
        smap = spatial_map(ic)
        im = ax.imshow(smap, cmap="coolwarm")
        #ax.contour(mask_crop.astype(np.uint8), colors="lime", linewidths=1)
        ax.set_title(f"Well {well} — epoch IC{ic}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    png = spatial_dir / f"well_{well}_epoch_spatialmaps_all.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    spatial_pngs.append(str(png))

    # -------------------------
    # Original ICA spatial-map energy
    # One row per retained IC, restricted to valid fish-mask pixels.
    # No Gaussian smoothing is applied here.
    # -------------------------
    normalized_energy_rows = []
    fish_mask = np.asarray(mask_crop, dtype=bool)

    for ic in range(n_keep_eff):
        smap = np.asarray(spatial_map(ic), dtype=np.float32)

        valid_mask = fish_mask & np.isfinite(smap)
        n_valid_pixels = int(valid_mask.sum())

        if n_valid_pixels > 0:
            valid_weights = smap[valid_mask].astype(np.float64, copy=False)
            original_energy = float(np.sum(valid_weights ** 2))
            original_normalized_energy = float(
                original_energy / n_valid_pixels
            )
        else:
            original_energy = np.nan
            original_normalized_energy = np.nan

        normalized_energy_rows.append({
            "well": str(well),
            "ic": int(ic),
            "n_valid_pixels": n_valid_pixels,
            "original_energy": original_energy,
            "original_normalized_energy": original_normalized_energy,
        })

    normalized_energy_csv_path = (
        ica_root / f"well_{well}_normalized_energy.csv"
    )

    pd.DataFrame(
        normalized_energy_rows,
        columns=[
            "well",
            "ic",
            "n_valid_pixels",
            "original_energy",
            "original_normalized_energy",
        ],
    ).to_csv(normalized_energy_csv_path, index=False)

    print(
        f"[INFO] Saved ICA normalized-energy CSV: "
        f"{normalized_energy_csv_path}"
    )

    # -------------------------
    # Blob feature extraction + debugging plots
    # -------------------------
    all_blob_rows = []
    trace_results = []
    fig_rows = []

    blob_csv_path = ica_root / f"well_{well}_epoch_blob_summary.csv"

    for ic in range(n_keep_eff):
        smap = spatial_map(ic)

        try:
            top_res = top_blobs_from_smap(
                smap=smap,
                mask_crop=mask_crop,
                percentile=90,
                use_abs=True,
                min_pixels=0,
                fill_holes=True,
                top_k=5,
            )
            fig_rows.append((ic, top_res, None))
        except Exception as e:
            fig_rows.append((ic, None, str(e)))
            all_blob_rows.append({
                "well": well,
                "ic": int(ic),
                "blob_rank": np.nan,
                "linearity_pca": np.nan,
                "max_amp_hilbert": np.nan,
                "is_periodic": np.nan,
                "blob_size": np.nan,
                "integral_abs": np.nan,
                "periodicity_score": np.nan,
                "best_freq_hz": np.nan,
                "best_bpm": np.nan,
                "rms": np.nan,
                "peak_signal_name": "",
                "n_peaks_all": np.nan,
                "n_peaks_real": np.nan,
                "interval_cv": np.nan,
                "fallback_motion_correction_used": int(motion_correction_used),
                "fallback_correction_sign": int(correction_sign),
                "signal_error": f"blob_detection_failed: {e}",
            })
            continue

        for blob in top_res["blobs"]:
            row = {
                "well": well,
                "ic": int(ic),
                "blob_rank": int(blob["rank"]),
                "linearity_pca": float(blob["linearity_pca"]) if np.isfinite(blob["linearity_pca"]) else np.nan,
                "max_amp_hilbert": np.nan,
                "is_periodic": np.nan,
                "blob_size": int(blob["blob_size"]),
                "integral_abs": np.nan,
                "periodicity_score": np.nan,
                "best_freq_hz": np.nan,
                "best_bpm": np.nan,
                "rms": np.nan,
                "peak_signal_name": "",
                "n_peaks_all": np.nan,
                "n_peaks_real": np.nan,
                "interval_cv": np.nan,
                "fallback_motion_correction_used": int(motion_correction_used),
                "fallback_correction_sign": int(correction_sign),
                "signal_error": "",
            }

            try:
                sig_res = reconstruct_and_analyze_blob_signal_epoch(
                    blob_mask=blob["blob_filled"],
                    videopath=videopath,
                    xoff=xoff,
                    yoff=yoff,
                    crop_size=crop_size,
                    fps_use=fps_use,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    low=0.7,
                    high=4.0,
                    order=3,
                    acf_periodicity_thresh=0.50,
                    min_lag_sec=0.25,
                    max_lag_sec=3.0,
                    prom_norm_thresh=0.40,
                    min_distance_sec=0.3,
                    camera_motion_df=camera_motion_df,
                    correction_sign=correction_sign,
                    epoch_gray_cube=epoch_gray_cube,
                    epoch_frames=epoch_frames,
                )

                linearity_pca_val = float(blob["linearity_pca"]) if np.isfinite(blob["linearity_pca"]) else np.nan
                blob_size_val = int(blob["blob_size"])
                max_amp_val = float(sig_res["max_amp_hilbert"])
                n_peaks_real = int(sig_res["n_peaks_real"])

                periodicity_val = float(sig_res["periodicity_score"])
                peak_signal_name = str(sig_res["peak_signal_name"])
                interval_cv_val = (
                    float(sig_res["interval_cv"])
                    if np.isfinite(sig_res["interval_cv"])
                    else np.nan
                )

                # No heartbeat accept/reject decision is made here.
                # These are raw signal/blob features used by XGBoost.
                row.update({
                    "max_amp_hilbert": float(sig_res["max_amp_hilbert"]),
                    "is_periodic": int(sig_res["is_periodic"]),
                    "integral_abs": float(sig_res["integral_abs"]),
                    "periodicity_score": float(sig_res["periodicity_score"]),
                    "best_freq_hz": float(sig_res["best_freq_hz"]) if np.isfinite(sig_res["best_freq_hz"]) else np.nan,
                    "best_bpm": float(sig_res["best_bpm"]) if np.isfinite(sig_res["best_bpm"]) else np.nan,
                    "rms": float(sig_res["rms"]),
                    "peak_signal_name": sig_res["peak_signal_name"],
                    "n_peaks_all": int(sig_res["n_peaks_all"]),
                    "n_peaks_real": int(sig_res["n_peaks_real"]),
                    "interval_cv": float(sig_res["interval_cv"]) if np.isfinite(sig_res["interval_cv"]) else np.nan,
                })

                trace_results.append({
                    "ic": int(ic),
                    "rank": int(blob["rank"]),
                    "blob_size": int(blob["blob_size"]),
                    "frames_blob": sig_res["frames_blob"],
                    "trace_raw": sig_res["trace_raw"],
                    "trace_filt": sig_res["trace_filt"],
                    "trace_hilbert": sig_res["trace_hilbert"],
                    "acf": sig_res["acf"],
                    "lags_sec": sig_res["lags_sec"],
                    "best_lag": sig_res["best_lag"],
                    "periodicity_score": sig_res["periodicity_score"],
                    "best_freq_hz": sig_res["best_freq_hz"],
                    "best_bpm": sig_res["best_bpm"],
                    "max_amp_hilbert": sig_res["max_amp_hilbert"],
                    "peak_signal_name": sig_res["peak_signal_name"],
                    "n_peaks_all": sig_res["n_peaks_all"],
                    "n_peaks_real": sig_res["n_peaks_real"],
                    "interval_cv": interval_cv_val,
                })

            except Exception as e:
                row["signal_error"] = f"signal_failed: {e}"

            all_blob_rows.append(row)

    # Save raw blob/signal measurements.
    # No row is accepted/rejected as heartbeat here.
    write_results_csv(all_blob_rows, blob_csv_path)

    # ---------------------------------------------------------
    # Build the exact 235-feature fish vector used for XGBoost
    # ---------------------------------------------------------
    fixed_235_df, fixed_235_cols = (
        build_fixed_235_feature_vector_from_fallback(
            all_blob_rows=all_blob_rows,
            normalized_energy_rows=normalized_energy_rows,
        )
    )

    fixed_235_csv_path = (
        ica_root / f"well_{well}_fish_fixed_235_features.csv"
    )
    fixed_235_feature_names_path = (
        ica_root / f"well_{well}_fish_fixed_235_feature_names.txt"
    )

    fixed_235_df.to_csv(fixed_235_csv_path, index=False)
    fixed_235_feature_names_path.write_text(
        "\n".join(fixed_235_cols) + "\n"
    )

    print(
        f"[INFO] Saved fallback 235-feature vector: "
        f"{fixed_235_csv_path}"
    )

    # ---------------------------------------------------------
    # Prepare the exact XGBoost input representation used by
    # optimized_grouped_ml_collapsed_NO_INTERVAL_CV.
    # No model inference is performed here yet.
    # ---------------------------------------------------------
    xgb_no_interval_cv_df, xgb_no_interval_cv_cols = (
        prepare_xgboost_no_interval_cv_features(
            fixed_235_df
        )
    )

    xgb_no_interval_cv_csv_path = (
        ica_root / f"well_{well}_xgb_no_interval_cv_features.csv"
    )
    xgb_no_interval_cv_feature_names_path = (
        ica_root / f"well_{well}_xgb_no_interval_cv_feature_names.txt"
    )

    xgb_no_interval_cv_df.to_csv(
        xgb_no_interval_cv_csv_path,
        index=False,
    )
    xgb_no_interval_cv_feature_names_path.write_text(
        "\n".join(xgb_no_interval_cv_cols) + "\n"
    )

    print(
        f"[INFO] Saved fallback XGBoost-ready feature vector "
        f"({len(xgb_no_interval_cv_cols)} predictors): "
        f"{xgb_no_interval_cv_csv_path}"
    )

    # ---------------------------------------------------------
    # XGBoost fallback inference
    # ---------------------------------------------------------
    xgb_prediction = predict_fallback_xgboost(
        X=xgb_no_interval_cv_df,
        model_path=xgb_model_path,
        mode=fallback_model_mode,
        alive_threshold=xgb_alive_threshold,
    )

    print(
        f"[INFO] well={well} XGBoost fallback: "
        f"mode={xgb_prediction['model_mode']}, "
        f"prediction={xgb_prediction['predicted_binary_label']}, "
        f"P(Alive)={xgb_prediction['p_alive']:.6f}, "
        f"P(Dead)={xgb_prediction['p_dead']:.6f}"
    )

    if xgb_prediction["model_mode"] == "collapsed":
        print(
            f"[INFO] well={well} collapsed detail: "
            f"class={xgb_prediction['predicted_multiclass_label']}, "
            f"P(Weak)={xgb_prediction['p_weak']:.6f}, "
            f"P(Strong)={xgb_prediction['p_strong']:.6f}"
        )
    # -------------------------
    # REPRESENTATIVE BLOB QC PNG
    # -------------------------
    # This image is for debugging/montage only.
    # It has NO role in Alive/Dead classification.
    #
    # Choose the row with:
    #   1) highest periodicity_score
    #   2) larger blob_size
    #   3) lower linearity_pca
    representative_rows = [
        row for row in all_blob_rows
        if np.isfinite(row.get("blob_rank", np.nan))
    ]

    if representative_rows:
        def _rep_key(row):
            periodicity = row.get("periodicity_score", np.nan)
            blob_size = row.get("blob_size", np.nan)
            linearity = row.get("linearity_pca", np.nan)

            periodicity = float(periodicity) if np.isfinite(periodicity) else -np.inf
            blob_size = float(blob_size) if np.isfinite(blob_size) else -np.inf
            linearity = float(linearity) if np.isfinite(linearity) else np.inf

            return (periodicity, blob_size, -linearity)

        representative = max(representative_rows, key=_rep_key)
        rep_ic = int(representative["ic"])
        rep_rank = int(representative["blob_rank"])

        rep_res = None
        for ic, res, err in fig_rows:
            if err is None and ic == rep_ic:
                for blob in res["blobs"]:
                    if int(blob["rank"]) == rep_rank:
                        rep_res = (res, blob)
                        break
            if rep_res is not None:
                break

        if rep_res is not None:
            res_qc, blob_qc = rep_res

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

            im0 = axes[0].imshow(res_qc["smap"], cmap="coolwarm")
            axes[0].contour(
                mask_crop.astype(np.uint8),
                colors="lime",
                linewidths=1,
            )
            axes[0].contour(
                blob_qc["blob_filled"].astype(np.uint8),
                colors="yellow",
                linewidths=2,
            )
            axes[0].set_title(
                f"QC IC{rep_ic} R{rep_rank} | "
                f"size={int(representative.get('blob_size', 0))} | "
                f"lin={representative.get('linearity_pca', np.nan):.3f} | "
                f"P={representative.get('periodicity_score', np.nan):.3f}"
            )
            axes[0].axis("off")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            axes[1].imshow(res_qc["binary_mask"], cmap="gray")
            axes[1].contour(
                blob_qc["blob_mask"].astype(np.uint8),
                colors="cyan",
                linewidths=2,
            )
            axes[1].set_title("Threshold mask")
            axes[1].axis("off")

            axes[2].imshow(blob_qc["blob_filled"], cmap="gray")
            axes[2].set_title("Representative blob")
            axes[2].axis("off")

            fig.tight_layout()

            # Keep the historical filename so the montage script still finds it.
            rep_png = blob_dir / f"well_{well}_epoch_blob_best.png"
            fig.savefig(rep_png, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # -------------------------
    # Blob summary figure PNG
    # -------------------------
    n_plot_rows = sum(
        len(res["blobs"]) if (res is not None) else 1
        for _, res, _ in fig_rows
    )

    if n_plot_rows > 0:
        fig, axes = plt.subplots(
            n_plot_rows, 3,
            figsize=(15, 4 * n_plot_rows),
            squeeze=False
        )

        row_idx = 0
        for ic, res, err in fig_rows:
            if err is not None:
                ax0, ax1, ax2 = axes[row_idx]
                for ax in (ax0, ax1, ax2):
                    ax.axis("off")
                    ax.text(
                        0.5, 0.5,
                        f"IC{ic}\nblob detection failed\n{err}",
                        ha="center", va="center", fontsize=11
                    )
                row_idx += 1
                continue

            for blob in res["blobs"]:
                ax0, ax1, ax2 = axes[row_idx]

                im0 = ax0.imshow(res["smap"], cmap="coolwarm")
                ax0.contour(mask_crop.astype(np.uint8), colors="lime", linewidths=1)
                ax0.contour(blob["blob_filled"].astype(np.uint8), colors="yellow", linewidths=1)
                ax0.set_title(f"IC{ic} blob rank {blob['rank']} signed map")
                ax0.axis("off")
                fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

                ax1.imshow(res["binary_mask"], cmap="gray")
                ax1.contour(blob["blob_mask"].astype(np.uint8), colors="cyan", linewidths=1)
                ax1.set_title(f"IC{ic} blob rank {blob['rank']} threshold mask")
                ax1.axis("off")

                ax2.imshow(blob["blob_filled"], cmap="gray")
                ax2.set_title(f"IC{ic} blob rank {blob['rank']} ({blob['blob_size']} px)")
                ax2.axis("off")

                row_idx += 1

        fig.tight_layout()
        blob_png = blob_dir / f"well_{well}_epoch_blob_summary.png"
        fig.savefig(blob_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -------------------------
    # Blob trace PNGs
    # -------------------------
    trace_pngs = []
    for tr in trace_results:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

        t_sec = tr["frames_blob"] / float(fps_use)

        axes[0].plot(t_sec, tr["trace_raw"], linewidth=1.0)
        axes[0].set_title(
            f"Well {well} IC{tr['ic']} blob rank {tr['rank']} raw trace"
        )
        axes[0].set_ylabel("Raw")

        axes[1].plot(t_sec, tr["trace_filt"], linewidth=1.0)
        axes[1].set_title(
            f"Bandpassed | size={tr['blob_size']} | periodicity={tr['periodicity_score']:.3f}"
        )
        axes[1].set_ylabel("Bandpassed")

        axes[2].plot(t_sec, tr["trace_hilbert"], linewidth=1.0)
        axes[2].set_title(
            f"Hilbert envelope | max_amp={tr['max_amp_hilbert']:.3f} | "
            f"n_peaks_real={tr['n_peaks_real']}"
        )
        axes[2].set_ylabel("Envelope")

        axes[3].plot(tr["lags_sec"], tr["acf"], linewidth=1.0)
        if tr["best_lag"] is not None:
            best_lag_sec = tr["best_lag"] / float(fps_use)
            axes[3].axvline(best_lag_sec, linestyle="--")
        axes[3].set_title(
            f"ACF | best_freq={tr['best_freq_hz']:.3f} Hz | best_bpm={tr['best_bpm']:.1f}"
            if np.isfinite(tr["best_freq_hz"]) and np.isfinite(tr["best_bpm"])
            else "ACF"
        )
        axes[3].set_xlabel("Time / Lag (s)")
        axes[3].set_ylabel("ACF")

        for ax in axes:
            ax.grid(alpha=0.2)

        fig.tight_layout()
        png = trace_dir / f"well_{well}_IC{tr['ic']:02d}_blobrank{tr['rank']:02d}_trace.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        trace_pngs.append(str(png))

    return {
        "time_signal_pngs": time_pngs,
        "spatial_map_pngs": spatial_pngs,
        "blob_summary_png": str(blob_dir / f"well_{well}_epoch_blob_summary.png"),
        "blob_trace_pngs": trace_pngs,
        "ic_summary_csv": str(blob_csv_path),
        "normalized_energy_csv": str(normalized_energy_csv_path),
        "fixed_235_feature_csv": str(fixed_235_csv_path),
        "fixed_235_feature_names_txt": str(fixed_235_feature_names_path),
        "fixed_235_feature_count": int(len(fixed_235_cols)),
        "xgb_no_interval_cv_feature_csv": str(xgb_no_interval_cv_csv_path),
        "xgb_no_interval_cv_feature_names_txt": str(xgb_no_interval_cv_feature_names_path),
        "xgb_no_interval_cv_feature_count": int(len(xgb_no_interval_cv_cols)),
        "xgb_model_mode": str(xgb_prediction["model_mode"]),
        "xgb_predicted_multiclass_int": xgb_prediction["predicted_multiclass_int"],
        "xgb_predicted_multiclass_label": str(
            xgb_prediction["predicted_multiclass_label"]
        ),
        "xgb_predicted_binary": int(xgb_prediction["predicted_binary"]),
        "xgb_predicted_binary_label": str(
            xgb_prediction["predicted_binary_label"]
        ),
        "xgb_p_dead": float(xgb_prediction["p_dead"]),
        "xgb_p_alive": float(xgb_prediction["p_alive"]),
        "xgb_p_weak": xgb_prediction["p_weak"],
        "xgb_p_strong": xgb_prediction["p_strong"],
        "xgb_alive_threshold": xgb_prediction["alive_threshold"],
        "n_ic_rows": int(len(all_blob_rows)),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "ica_converged": int(ica_converged),
        "ica_n_iter": int(ica_n_iter) if np.isfinite(ica_n_iter) else np.nan,
        "ica_max_iter": int(ica_max_iter),
        "motion_correction_used": int(motion_correction_used),
        "correction_sign": int(correction_sign),
    }
def mask_overlap_fraction_over_smaller(mask1, mask2):
    if mask1 is None or mask2 is None:
        return 0.0

    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)

    if m1.shape != m2.shape:
        raise ValueError(f"Mask shapes do not match: {m1.shape} vs {m2.shape}")

    a1 = int(m1.sum())
    a2 = int(m2.sum())
    if a1 == 0 or a2 == 0:
        return 0.0

    inter = int((m1 & m2).sum())
    return float(inter / min(a1, a2))
# =========================================================
# I/O helpers
# =========================================================
def well_list():
    return [f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)]


def get_well_from_array_id(task_id):
    wells = well_list()
    if task_id < 1 or task_id > len(wells):
        raise ValueError(f"SLURM_ARRAY_TASK_ID must be 1..{len(wells)}")
    return wells[task_id - 1]


def find_video_for_well(plate_dir, well):
    patterns = [
        f"well_{well}.mp4",
        f"{well}.mp4",
        f"well_{well}.avi",
        f"{well}.avi",
        f"well_{well}.mov",
        f"{well}.mov",
    ]
    plate_dir = Path(plate_dir)
    for p in patterns:
        cand = plate_dir / p
        if cand.exists():
            return str(cand)

    matches = list(plate_dir.glob(f"*{well}*.mp4"))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(f"Could not find video for well {well} in {plate_dir}")


def save_summary_row(csv_path, row_dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def _save_json_safe(path: Path, payload: dict, prefix: str = "[INFO]"):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"{prefix} saved reusable JSON: {path}")
    except Exception as e:
        print(f"[WARN] failed to save JSON {path}: {e}")


def _save_npz_safe(path: Path, prefix: str = "[INFO]", **arrays):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        print(f"{prefix} saved reusable NPZ: {path}")
    except Exception as e:
        print(f"[WARN] failed to save NPZ {path}: {e}")



def pad_bool_mask(mask, pad_frames=7):
    """
    Expand True frames by +/- pad_frames.
    Frame-to-frame absdiff artifacts usually affect neighboring frames too.
    """
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()

    bad = np.where(mask)[0]
    for idx in bad:
        lo = max(0, int(idx) - int(pad_frames))
        hi = min(len(mask), int(idx) + int(pad_frames) + 1)
        out[lo:hi] = True

    return out


def outside_motion_veto_mask(
    outside_trace,
    k_mad=10.0,
    q=0.995,
    pad_frames=7,
):
    """
    Detect plate/camera motion from outside pixels.

    Conservative threshold:
        threshold = max(median + k_mad * robust_sigma, q-quantile)

    This keeps normal small outside pulses but discards unusually large
    outside-motion events from the T2 decision.
    """
    x = np.asarray(outside_trace, dtype=float)
    veto = np.zeros_like(x, dtype=bool)

    good = np.isfinite(x)
    if int(good.sum()) < 10:
        info = {
            "outside_median": np.nan,
            "outside_mad": np.nan,
            "outside_threshold": np.nan,
            "outside_quantile_threshold": np.nan,
            "outside_robust_threshold": np.nan,
            "n_veto_raw": 0,
            "n_veto_padded": 0,
        }
        return veto, info

    vals = x[good]

    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med))) + 1e-12
    robust_sigma = 1.4826 * mad

    robust_thr = med + float(k_mad) * robust_sigma
    quant_thr = float(np.nanquantile(vals, float(q)))

    thr = max(float(robust_thr), float(quant_thr))

    veto_raw = x >= thr
    veto = pad_bool_mask(veto_raw, pad_frames=pad_frames)

    info = {
        "outside_median": med,
        "outside_mad": float(mad),
        "outside_threshold": float(thr),
        "outside_quantile_threshold": float(quant_thr),
        "outside_robust_threshold": float(robust_thr),
        "n_veto_raw": int(veto_raw.sum()),
        "n_veto_padded": int(veto.sum()),
    }

    return veto, info


# =========================================================
# Main
# =========================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plate_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--well", default=None)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=0)

    # fallback epoch trigger
    parser.add_argument("--fallback_min_epoch_sec", type=float, default=5.0)

    # XGBoost fallback classifier
    parser.add_argument(
        "--xgb_model_path",
        type=str,
        default=DEFAULT_XGB_MODEL_PATH,
        help=(
            "Path to a saved XGBClassifier model (.json/.ubj). "
            "Required when a no-motion well reaches fallback inference."
        ),
    )
    parser.add_argument(
        "--fallback_model_mode",
        type=str,
        default=DEFAULT_FALLBACK_MODEL_MODE,
        choices=["binary", "collapsed"],
        help=(
            "binary: dedicated Dead/Alive model. "
            "collapsed: Dead/Weak/Strong model collapsed to Dead/Alive."
        ),
    )
    parser.add_argument(
        "--xgb_alive_threshold",
        type=float,
        default=DEFAULT_XGB_ALIVE_THRESHOLD,
        help=(
            "Alive probability threshold for binary mode only. "
            "Collapsed mode follows multiclass argmax exactly."
        ),
    )

    # fixed center crop
    parser.add_argument("--crop_size", type=int, default=1048)

    # SAM fallback config
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Override SAM_CHECKPOINT or the bundled checkpoint location.",
    )
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--lower_area", type=int, default=8000)
    parser.add_argument("--upper_area", type=int, default=35000)
    parser.add_argument("--search_max", type=int, default=60)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--dilate_px", type=int, default=10)
    parser.add_argument("--min_size", type=int, default=50)

    # epoch ICA config
    parser.add_argument("--ica_n_components", type=int, default=3)
    parser.add_argument("--ica_random_state", type=int, default=32)
    parser.add_argument("--ica_fun", type=str, default="exp")
    parser.add_argument("--ica_max_iter", type=int, default=1000)
    parser.add_argument("--ica_tol", type=float, default=1e-4)

    parser.add_argument(
        "--reusable_parent_dir",
        type=str,
        default="",
        help=(
            "Optional parent folder for reusable artifacts (ring params, masks, "
            "fallback union mask/metadata) across reruns and downstream scripts."
        ),
    )

    # Profiling
    parser.add_argument(
        "--profile",
        type=int,
        default=1,
        choices=[0, 1],
        help="When 1, automatically save cProfile outputs into out_dir/profiling.",
    )
    parser.add_argument(
        "--profile_top_n",
        type=int,
        default=200,
        help="How many functions to include in text cProfile reports.",
    )
    parser.add_argument(
        "--save_optional_qc",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "When 1, save optional QC overlays/CSVs (frame0 circle crop, "
            "inside-ring-outside overlays, plate-edge overlays/mask PNG, "
            "inside-outside timeseries/epochs CSV)."
        ),
    )
    parser.add_argument(
        "--save_sam_debug_masks",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "When 1, save SAM debug artifacts under debug_masks_area_filtered "
            "during fallback mask search."
        ),
    )

    args = parser.parse_args()

    # Validate the model configured in this Python file.
    if not Path(args.xgb_model_path).exists():
        raise FileNotFoundError(
            f"Configured XGBoost model does not exist: "
            f"{args.xgb_model_path}"
        )

    print(
        f"[INFO] XGBoost fallback configured: "
        f"mode={args.fallback_model_mode}, "
        f"model={args.xgb_model_path}"
    )
    save_optional_qc = bool(int(args.save_optional_qc))
    save_sam_debug_masks = bool(int(args.save_sam_debug_masks))

    if args.well is None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        well = get_well_from_array_id(task_id)
    else:
        well = args.well

    plate_dir = Path(args.plate_dir)
    plate_name = plate_dir.name
    out_dir = Path(args.out_root) / plate_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reusable_well_dir = None
    if str(args.reusable_parent_dir).strip() != "":
        reusable_well_dir = (
            Path(args.reusable_parent_dir)
            / plate_name
            / f"well_{well}"
        )
        reusable_well_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] reusable artifact dir: {reusable_well_dir}")

    _start_auto_profile(
        out_dir=out_dir,
        well=well,
        enabled=bool(int(args.profile)),
        top_n=int(args.profile_top_n),
    )

    videopath = find_video_for_well(plate_dir, well)
    max_frames = None if args.max_frames == 0 else args.max_frames

    # ---------- load frame 0 ----------
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    ret, frame0 = cap.read()
    if not ret or frame0 is None:
        cap.release()
        raise RuntimeError(f"Could not read frame 0 from: {videopath}")

    frame_h, frame_w = frame0.shape[:2]
    rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # ---------- detect/reuse circle ----------
    reused_ring_params = 0
    ring_params_path = (
        reusable_well_dir / "ring_params.json"
        if reusable_well_dir is not None else None
    )

    res = None

    if ring_params_path is not None and ring_params_path.exists():
        try:
            with open(ring_params_path, "r") as f:
                ring_cache = json.load(f)

            cached_h = int(ring_cache.get("frame_h", frame_h))
            cached_w = int(ring_cache.get("frame_w", frame_w))

            cx_cached = float(ring_cache["circle_cx"])
            cy_cached = float(ring_cache["circle_cy"])
            r_cached = float(ring_cache["circle_radius"])
            rmse_cached = float(ring_cache.get("circle_rmse", np.nan))

            if (
                cached_h == int(frame_h)
                and cached_w == int(frame_w)
                and np.isfinite(cx_cached)
                and np.isfinite(cy_cached)
                and np.isfinite(r_cached)
                and r_cached > 0
            ):
                res = {
                    "cx": cx_cached,
                    "cy": cy_cached,
                    "radius": r_cached,
                    "rmse": rmse_cached,
                    "p_dark": -1,
                    "thr": np.nan,
                    "component_mask": np.zeros((frame_h, frame_w), dtype=bool),
                    "ring_mask": np.zeros((frame_h, frame_w), dtype=bool),
                    "score": np.nan,
                }
                reused_ring_params = 1
                print(f"[INFO] reused ring params from: {ring_params_path}")

        except Exception as e:
            print(f"[WARN] failed to reuse ring params ({ring_params_path}): {e}")

    if res is None:
        res = detect_ring_auto_prefer_annulus(
            rgb,
            p_range=range(6, 16),
            morph_ksize=7,
            min_area=100000,
            max_fill_ratio=0.75,
            min_ring_pixels=200,
            min_coverage_rad=1.5 * np.pi,
            inner_band_low=2.0,
            inner_band_high=6.0,
            expected_center=(rgb.shape[1] / 2.0, rgb.shape[0] / 2.0),
            center_weight=0.002,
        )

        if ring_params_path is not None:
            _save_json_safe(
                ring_params_path,
                {
                    "well": well,
                    "plate": plate_name,
                    "frame_h": int(frame_h),
                    "frame_w": int(frame_w),
                    "circle_cx": float(res["cx"]),
                    "circle_cy": float(res["cy"]),
                    "circle_radius": float(res["radius"]),
                    "circle_rmse": float(res["rmse"]),
                },
            )

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if np.isfinite(res.get("thr", np.nan)):
        binary = (gray <= res["thr"])
    else:
        binary = np.zeros_like(gray, dtype=bool)

    save_ring_debug_plot(
        rgb=rgb,
        gray=gray,
        binary=binary,
        comp_mask=res["component_mask"],
        ring_mask=res["ring_mask"],
        res=res,
        out_path=out_dir / f"well_{well}_ring_debug.png",
    )
    # We do NOT use the annulus (R to R+delta_R) for background subtraction
    # because it can contain fish reflections near the well edge.
    # Instead, we leave it as a buffer zone and use only:
    #   - inside circle for signal
    #   - far outside (beyond R+delta_R) for background

    # ---------- circle masks for inside/outside signal ----------
    # H, W = rgb.shape[:2]
    # yy, xx = np.ogrid[:H, :W]
    # dist2 = (xx - res["cx"]) ** 2 + (yy - res["cy"]) ** 2
    # r2 = res["radius"] ** 2

    # mask_inside_circle = dist2 <= r2
    # mask_outside_circle = dist2 > r2
    H, W = rgb.shape[:2]
    yy, xx = np.ogrid[:H, :W]

    cx, cy = res["cx"], res["cy"]
    R = res["radius"]

    # user control: how much larger the "buffer" zone should be
    delta_R = 200   # pixels; adjust as needed
    #R_outer = R + delta_R
    R_outer = min(R + delta_R, 600)  # cap the outer radius to avoid going too far

    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

    # masks (reuse from cache if available and valid)
    reused_region_masks = 0
    region_masks_path = (
        reusable_well_dir / "region_masks.npz"
        if reusable_well_dir is not None else None
    )
    region_masks_meta_path = (
        reusable_well_dir / "region_masks_metadata.json"
        if reusable_well_dir is not None else None
    )

    mask_inside_circle = None
    mask_ring = None
    mask_outside_circle = None
    plate_control_mask = None
    plate_control_info = None

    if region_masks_path is not None and region_masks_path.exists():
        try:
            with np.load(region_masks_path) as region_npz:
                inside_loaded = region_npz["inside_mask"].astype(bool)
                ring_loaded = region_npz["ring_mask"].astype(bool)
                outside_loaded = region_npz["outside_mask"].astype(bool)
                plate_loaded = region_npz["plate_control_mask"].astype(bool)

            same_shape = (
                inside_loaded.shape == (H, W)
                and ring_loaded.shape == (H, W)
                and outside_loaded.shape == (H, W)
                and plate_loaded.shape == (H, W)
            )

            if same_shape and int(inside_loaded.sum()) > 0 and int(outside_loaded.sum()) > 0:
                mask_inside_circle = inside_loaded
                mask_ring = ring_loaded
                mask_outside_circle = outside_loaded
                plate_control_mask = plate_loaded
                reused_region_masks = 1
                print(f"[INFO] reused region masks from: {region_masks_path}")

                meta_payload = {}
                if region_masks_meta_path is not None and region_masks_meta_path.exists():
                    with open(region_masks_meta_path, "r") as f:
                        meta_payload = json.load(f)

                plate_control_info = {
                    "edge_percentile": float(meta_payload.get("edge_percentile", 85.0)),
                    "min_edge_pixels": int(meta_payload.get("min_edge_pixels", 800)),
                    "n_edge_pixels": int(plate_control_mask.sum()),
                    "used_fallback_outer_ring": int(meta_payload.get("used_fallback_outer_ring", 0)),
                }

        except Exception as e:
            print(f"[WARN] failed to reuse region masks ({region_masks_path}): {e}")

    if mask_inside_circle is None:
        mask_inside_circle = dist2 <= R**2
        mask_ring = (dist2 > R**2) & (dist2 <= R_outer**2)
        mask_outside_circle = dist2 > R_outer**2

    # ---------- optional QC overlays for mask regions ----------
    if save_optional_qc:
        save_inside_ring_outside_overlay(
            rgb=rgb,
            mask_inside=mask_inside_circle,
            mask_ring=mask_ring,
            mask_outside=mask_outside_circle,
            out_path=out_dir / f"well_{well}_inside_ring_outside_overlay.png",
            title=f"{well}: inside / ring / outside overlay",
        )

        save_mask_overlay(
            rgb=rgb,
            mask=mask_outside_circle,
            out_path=out_dir / f"well_{well}_outside_only_overlay.png",
            title=f"{well}: outside-only region",
            color=(0, 120, 255),
            alpha=0.28,
        )

    # ---------- plate/camera motion control mask ----------
    # Use strong edge pixels from the far outside region.
    # If the outside is too bright/flat and has too few edges, the helper
    # adds only the outermost part of the ring/buffer, not the full ring.
    if plate_control_mask is None:
        plate_control_mask, plate_control_info = make_plate_edge_control_mask(
            rgb=rgb,
            mask_outside_circle=mask_outside_circle,
            mask_ring=mask_ring,
            dist2=dist2,
            R=R,
            R_outer=R_outer,
            edge_percentile=85,
            min_edge_pixels=800,
        )

        if reusable_well_dir is not None:
            _save_npz_safe(
                reusable_well_dir / "region_masks.npz",
                inside_mask=mask_inside_circle.astype(np.uint8),
                ring_mask=mask_ring.astype(np.uint8),
                outside_mask=mask_outside_circle.astype(np.uint8),
                plate_control_mask=plate_control_mask.astype(np.uint8),
            )

            _save_json_safe(
                reusable_well_dir / "region_masks_metadata.json",
                {
                    "well": well,
                    "plate": plate_name,
                    "edge_percentile": float(plate_control_info["edge_percentile"]),
                    "min_edge_pixels": int(plate_control_info["min_edge_pixels"]),
                    "used_fallback_outer_ring": int(plate_control_info["used_fallback_outer_ring"]),
                },
            )

    if save_optional_qc:
        save_mask_overlay(
            rgb=rgb,
            mask=plate_control_mask,
            out_path=out_dir / f"well_{well}_plate_edge_control_overlay.png",
            title=(
                f"{well}: plate edge-control mask "
                f"(n={plate_control_info['n_edge_pixels']}, "
                f"fallback_outer_ring={plate_control_info['used_fallback_outer_ring']})"
            ),
            color=(255, 0, 255),
            alpha=0.50,
        )

        plt.imsave(
            out_dir / f"well_{well}_plate_edge_control_mask.png",
            plate_control_mask.astype(np.uint8),
            cmap="gray",
        )

    # ---------- compute all motion traces in ONE video pass ----------
    # Inside/outside drive the fitted residual.
    # The edge-control trace is QC/artifact-veto only.
    res_in, res_out, res_plate_edge = compute_motion_traces_single_pass(
        videopath=videopath,
        mask_inside=mask_inside_circle,
        mask_outside=mask_outside_circle,
        mask_edge=plate_control_mask,
        downscale=args.downscale,
        max_frames=max_frames,
    )

    n_in = int(mask_inside_circle.sum())
    n_out = int(mask_outside_circle.sum())

    in_norm = res_in["motion"] / max(n_in, 1)
    out_norm = res_out["motion"] / max(n_out, 1)

    n_plate_edge_ds = downscaled_mask_count(
        plate_control_mask,
        downscale=args.downscale,
    )
    plate_edge_norm = res_plate_edge["motion"] / max(n_plate_edge_ds, 1)

    motion_sub = in_norm - out_norm
    t = res_in["time_sec"]

    good = np.isfinite(motion_sub) & np.isfinite(t)
    motion_sub = motion_sub[good]
    t = t[good]
    valid_frame_idxs = res_in["frame_idxs"][good]
    valid_in_norm = in_norm[good]
    valid_out_norm = out_norm[good]

    # Align plate-edge trace to the same frame indices used by inside/outside.
    plate_edge_by_frame = {
        int(f): float(v)
        for f, v in zip(res_plate_edge["frame_idxs"], plate_edge_norm)
    }
    valid_plate_edge_norm = np.array(
        [plate_edge_by_frame.get(int(f), np.nan) for f in valid_frame_idxs],
        dtype=float,
    )

    # ==========================================================
    # FITTED INSIDE-OUTSIDE RESIDUAL
    # ==========================================================
    # Fit outside -> inside, then subtract fitted outside.
    # This is better than simple inside - outside because inside and outside
    # regions usually respond with different amplitudes to common plate motion.
    A = np.c_[valid_out_norm, np.ones_like(valid_out_norm)]
    beta, alpha = np.linalg.lstsq(A, valid_in_norm, rcond=None)[0]
    beta = float(beta)
    alpha = float(alpha)

    fitted_inside_background = beta * valid_out_norm + alpha
    motion_sub = valid_in_norm - fitted_inside_background

    # Full-length fitted residual for diagnostic plot
    motion_sub_plot = in_norm - (beta * out_norm + alpha)

    save_inside_outside_diagnostic_plot(
        res_in=res_in,
        res_out=res_out,
        in_norm=in_norm,
        out_norm=out_norm,
        motion_sub=motion_sub_plot,
        well=well,
        out_path=out_dir / f"well_{well}_inside_outside_diagnostic_fitted.png",
        beta=float(beta),
        alpha=float(alpha),
    )

    if motion_sub.size < 2:
        raise RuntimeError(f"Too few valid motion points for well {well}")

    # ==========================================================
    # OUTSIDE-MOTION PLATE/CAMERA VETO
    # ==========================================================
    # Use outside pixels as the artifact detector.
    # If outside motion is unusually high, the frame is likely contaminated
    # by plate/camera motion and is discarded from T2 calls.
    # Raw outside veto: works when the outside region has usable brightness texture.
    outside_artifact, outside_veto_info = outside_motion_veto_mask(
        valid_out_norm,
        k_mad=10.0,
        q=0.995,
        pad_frames=7,
    )

    # Edge-control veto: works better when the outside is bright/flat.
    # It uses strong edge pixels in the far outside region, with fallback to only
    # the outermost ring/buffer if outside has too few edges.
    edge_artifact, edge_veto_info = outside_motion_veto_mask(
        valid_plate_edge_norm,
        k_mad=10.0,
        q=0.995,
        pad_frames=7,
    )

    # Final plate/camera artifact mask
    # EDGE-ONLY VETO:
    # Use the edge-control trace for T2 artifact rejection.
    # The outside brightness veto is still computed and saved for QC only,
    # but it does not discard frames.
    plate_artifact = edge_artifact.copy()

    veto_info = {
        "outside_threshold": outside_veto_info["outside_threshold"],
        "outside_raw_veto": outside_veto_info["n_veto_raw"],
        "outside_padded_veto": outside_veto_info["n_veto_padded"],
        "edge_threshold": edge_veto_info["outside_threshold"],
        "edge_raw_veto": edge_veto_info["n_veto_raw"],
        "edge_padded_veto": edge_veto_info["n_veto_padded"],
        "final_padded_veto": int(plate_artifact.sum()),
    }

    print(
        f"[INFO] EDGE-ONLY plate veto: "
        f"edge_threshold={veto_info['edge_threshold']:.6g}, "
        f"edge_raw_veto={veto_info['edge_raw_veto']}, "
        f"final_edge_padded_veto={veto_info['final_padded_veto']}; "
        f"outside_threshold_QC_only={veto_info['outside_threshold']:.6g}, "
        f"outside_raw_veto_QC_only={veto_info['outside_raw_veto']}"
    )

    # This is the signal used for T2 decisions.
    # The original fitted residual is still plotted, but artifact frames are
    # ignored for baseline and positive-deviation calls.
    motion_sub_for_t2 = motion_sub.copy()
    motion_sub_for_t2[plate_artifact] = np.nan

    # ---------- thresholds / states / epochs ----------
    motion_valid_for_t2 = motion_sub_for_t2[np.isfinite(motion_sub_for_t2)]

    if motion_valid_for_t2.size < 5:
        raise RuntimeError(f"Too few non-artifact frames for T2 in well {well}")

    counts, edges = np.histogram(motion_valid_for_t2, bins=100)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode_val = float(centers[np.argmax(counts)])

    mad = np.nanmedian(np.abs(motion_valid_for_t2 - mode_val))
    sigma_robust = float(1.4826 * mad)
    if not np.isfinite(sigma_robust) or sigma_robust <= 0:
        sigma_robust = float(np.nanstd(motion_valid_for_t2, ddof=1)) if motion_valid_for_t2.size > 1 else 0.0

    # ---------------------------------------------------------
    # baseline from central band around the mode
    # ---------------------------------------------------------
    baseline_mask = (
        np.isfinite(motion_sub_for_t2)
        & (np.abs(motion_sub_for_t2 - mode_val) <= 1.0 * sigma_robust)
    )
    x_base = motion_sub_for_t2[baseline_mask]

    if x_base.size < 5:
        finite_idx = np.where(np.isfinite(motion_sub_for_t2))[0]
        fallback_n = min(max(5, int(0.2 * len(finite_idx))), len(finite_idx))

        order = np.argsort(np.abs(motion_sub_for_t2[finite_idx] - mode_val))
        chosen = finite_idx[order[:fallback_n]]

        x_base = motion_sub_for_t2[chosen]
        baseline_mask = np.zeros_like(motion_sub, dtype=bool)
        baseline_mask[chosen] = True

    mu_base = float(np.nanmean(x_base))
    sd_base = float(np.nanstd(x_base, ddof=1)) if x_base.size > 1 else 0.0

    # ---------------------------------------------------------
    # positive deviation from baseline only
    # ---------------------------------------------------------
    pos_dev = motion_sub_for_t2 - mu_base
    #POS_DEV_THRESHOLD = 0.013
    POS_DEV_THRESHOLD = 0.021

    # ==========================================================
    # MOTION PEAK TIERS
    # ==========================================================
    # Tier 1: at least one local positive-deviation peak >= 0.021
    #         -> alive_by_motion
    #
    # Tier 2: at least two local positive-deviation peaks in
    #         [0.010, 0.021)
    #         -> alive_by_slight_motion
    #
    # Edge-vetoed samples are NaN in pos_dev and are not counted.
    # "Multiple" is defined as two or more peaks.
    # ==========================================================
    STRONG_MOTION_PEAK_THRESHOLD = POS_DEV_THRESHOLD
    SLIGHT_MOTION_PEAK_LOW = 0.010
    MIN_SLIGHT_MOTION_PEAKS = 2

    pos_dev_for_peak_search = np.where(
        np.isfinite(pos_dev),
        pos_dev,
        -np.inf,
    )

    motion_peak_indices, motion_peak_props = signal.find_peaks(
        pos_dev_for_peak_search,
        height=SLIGHT_MOTION_PEAK_LOW,
    )

    motion_peak_heights = np.asarray(
        motion_peak_props.get("peak_heights", np.array([], dtype=float)),
        dtype=float,
    )

    strong_motion_peak_mask = (
        motion_peak_heights >= STRONG_MOTION_PEAK_THRESHOLD
    )
    slight_motion_peak_mask = (
        (motion_peak_heights >= SLIGHT_MOTION_PEAK_LOW)
        & (motion_peak_heights < STRONG_MOTION_PEAK_THRESHOLD)
    )

    strong_motion_peak_indices = motion_peak_indices[strong_motion_peak_mask]
    slight_motion_peak_indices = motion_peak_indices[slight_motion_peak_mask]

    n_strong_motion_peaks = int(len(strong_motion_peak_indices))
    n_slight_motion_peaks = int(len(slight_motion_peak_indices))

    mobility_alive_by_strong_motion = int(
        n_strong_motion_peaks >= 1
    )
    mobility_alive_by_slight_motion = int(
        mobility_alive_by_strong_motion == 0
        and n_slight_motion_peaks >= MIN_SLIGHT_MOTION_PEAKS
    )

    mobility_alive_by_motion_tier = int(
        mobility_alive_by_strong_motion
        or mobility_alive_by_slight_motion
    )

    if mobility_alive_by_strong_motion:
        mobility_motion_tier = "alive_by_motion"
    elif mobility_alive_by_slight_motion:
        mobility_motion_tier = "alive_by_slight_motion"
    else:
        mobility_motion_tier = "no_motion_call"

    print(
        f"[INFO] well={well} motion tiers: "
        f"strong_peaks={n_strong_motion_peaks} "
        f"(>= {STRONG_MOTION_PEAK_THRESHOLD:.3f}), "
        f"slight_peaks={n_slight_motion_peaks} "
        f"([{SLIGHT_MOTION_PEAK_LOW:.3f}, "
        f"{STRONG_MOTION_PEAK_THRESHOLD:.3f})), "
        f"tier={mobility_motion_tier}"
    )

    # ==========================================================
    # TWO STATE ARRAYS
    # ==========================================================
    # state_t2:
    #   Used for T2 plot/CSV.
    #   Edge-veto frames are shown as -1 / gray.
    #
    # state_epoch:
    #   Used ONLY for longest_baseline_epoch selection.
    #   Edge-veto frames do NOT divide/split the epoch.
    #
    # Result:
    #   gray edge-veto frames are ignored for T2 calls,
    #   but they do not break the fallback baseline epoch.
    # ==========================================================

    state_t2 = np.zeros_like(motion_sub, dtype=int)

    # -1 = discarded plate/camera artifact frame for T2 visualization only
    state_t2[plate_artifact] = -1

    # Only non-artifact frames can become positive fish-motion T2 events
    state_t2[np.isfinite(pos_dev) & (pos_dev >= POS_DEV_THRESHOLD)] = 1

    # For longest-baseline/fallback epoch selection, do NOT let edge-veto
    # frames split the baseline. Convert -1 artifact frames back to 0.
    state_epoch = state_t2.copy()
    state_epoch[state_epoch == -1] = 0

    # Epochs used for T2 plot / CSV
    epochs = get_state_epochs(state_t2, t, valid_frame_idxs)
    above_epochs = [ep for ep in epochs if ep["state"] == 1]
    below_epochs = [ep for ep in epochs if ep["state"] == -1]
    baseline_epochs_for_plot = [ep for ep in epochs if ep["state"] == 0]

    # Epochs used for fallback selection
    epoch_selection_epochs = get_state_epochs(state_epoch, t, valid_frame_idxs)
    baseline_epochs = [ep for ep in epoch_selection_epochs if ep["state"] == 0]

    # Keep downstream variable name for CSV/summary compatibility.
    # This remains the T2/display state, not the fallback-selection state.
    state = state_t2

    finite_pos_idx = np.where(np.isfinite(pos_dev))[0]
    if finite_pos_idx.size > 0:
        max_pos_dev_idx = int(finite_pos_idx[np.argmax(pos_dev[finite_pos_idx])])
        max_pos_dev_from_baseline = float(pos_dev[max_pos_dev_idx])
        max_pos_dev_signal = float(motion_sub[max_pos_dev_idx])
        max_pos_dev_time = float(t[max_pos_dev_idx])
    else:
        max_pos_dev_idx = -1
        max_pos_dev_from_baseline = np.nan
        max_pos_dev_signal = np.nan
        max_pos_dev_time = np.nan

    if len(baseline_epochs) > 0:
        longest_baseline_epoch = max(baseline_epochs, key=lambda ep: ep["duration_sec"])
    else:
        longest_baseline_epoch = None



    # ---------- detected-center 1048 crop ----------
    crop_xoff, crop_yoff = center_crop_origin_from_detected_center(
        cx=res["cx"],
        cy=res["cy"],
        frame_w=frame_w,
        frame_h=frame_h,
        crop_size=args.crop_size,
    )

    crop_mask_full = crop_rect_mask(
        frame_h=frame_h,
        frame_w=frame_w,
        xoff=crop_xoff,
        yoff=crop_yoff,
        crop_size=args.crop_size,
    )

    if save_optional_qc:
        # plot crop on frame 0
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        circle = plt.Circle((res["cx"], res["cy"]), res["radius"], color="cyan", fill=False, linewidth=2.5)
        ax.add_patch(circle)
        rect = plt.Rectangle(
            (crop_xoff, crop_yoff),
            args.crop_size,
            args.crop_size,
            fill=False,
            color="yellow",
            linewidth=2.5,
        )
        ax.add_patch(rect)
        ax.plot(res["cx"], res["cy"], "bo", markersize=5)
        ax.set_title(
            f"{well} frame0\n"
            f"circle center=({res['cx']:.1f}, {res['cy']:.1f}), r={res['radius']:.1f}\n"
            f"1048 crop origin=({crop_xoff}, {crop_yoff})"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"well_{well}_frame0_circle_and_1048crop.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---------- plot inside-outside signal ----------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, motion_sub, linewidth=1.0, label="inside - fitted outside")

    ymin = float(np.min(motion_sub))
    ymax = float(np.max(motion_sub))

    ax.fill_between(
        t, ymin, ymax,
        where=plate_artifact,
        alpha=0.22,
        color="gray",
        label="edge-motion veto"
    )

    ax.fill_between(
        t, ymin, ymax,
        where=(np.isfinite(pos_dev) & (pos_dev >= POS_DEV_THRESHOLD)),
        alpha=0.18,
        color="red",
        label=f"pos_dev >= {POS_DEV_THRESHOLD:.3f}"
    )

    ax.axhline(mu_base, linestyle="--", linewidth=1.2, alpha=0.9, color="black", label="baseline_mean")
    ax.axhline(mu_base + POS_DEV_THRESHOLD, linestyle="--", linewidth=1.5, alpha=0.9, color="red", label="baseline + threshold")
    if np.isfinite(max_pos_dev_time) and np.isfinite(max_pos_dev_signal):
        ax.scatter(
            [max_pos_dev_time],
            [max_pos_dev_signal],
            s=28,
            color="magenta",
            zorder=5,
            label="max positive deviation",
        )

    if n_strong_motion_peaks > 0:
        ax.scatter(
            t[strong_motion_peak_indices],
            motion_sub[strong_motion_peak_indices],
            s=34,
            marker="^",
            color="red",
            zorder=6,
            label="strong motion peak",
        )

    if n_slight_motion_peaks > 0:
        ax.scatter(
            t[slight_motion_peak_indices],
            motion_sub[slight_motion_peak_indices],
            s=30,
            marker="o",
            facecolors="none",
            edgecolors="orange",
            linewidths=1.5,
            zorder=6,
            label="slight motion peak",
        )

    txt_lines = [
        f"mode: {mode_val:.6f}",
        f"baseline_mean: {mu_base:.6f}",
        f"strong peak threshold: {STRONG_MOTION_PEAK_THRESHOLD:.6f}",
        f"slight peak range: [{SLIGHT_MOTION_PEAK_LOW:.3f}, {STRONG_MOTION_PEAK_THRESHOLD:.3f})",
        f"strong peaks: {n_strong_motion_peaks}",
        f"slight peaks: {n_slight_motion_peaks}",
        f"motion tier: {mobility_motion_tier}",
        f"max_pos_dev: {max_pos_dev_from_baseline:.6f}",
    ]
    if longest_baseline_epoch is not None:
        txt_lines.append(f"longest baseline: {longest_baseline_epoch['duration_sec']:.3f} s")
        txt_lines.append(f"frames: {longest_baseline_epoch['start_frame']}–{longest_baseline_epoch['end_frame']}")

    ax.text(
        0.99, 0.98,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inside - fitted outside")
    ax.set_title(f"{well}: positive deviation from baseline")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"well_{well}_inside_minus_outside_t2_epochs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- epoch-only fallback on longest baseline epoch ----------
    used_longest_epoch_fallback = 0
    longest_epoch_fallback_reason = ""
    epoch_union_mask_npy = ""
    epoch_ica_summary_csv = ""
    epoch_ica_time_pngs = []
    epoch_ica_spatial_pngs = []
    epoch_blob_summary_png = ""
    epoch_blob_trace_pngs = []
    fallback_fixed_235_feature_csv = ""
    fallback_fixed_235_feature_names_txt = ""
    fallback_fixed_235_feature_count = 0
    fallback_xgb_no_interval_cv_feature_csv = ""
    fallback_xgb_no_interval_cv_feature_names_txt = ""
    fallback_xgb_no_interval_cv_feature_count = 0

    fallback_xgb_model_mode = ""
    fallback_xgb_predicted_multiclass_int = np.nan
    fallback_xgb_predicted_multiclass_label = ""
    fallback_xgb_predicted_binary = np.nan
    fallback_xgb_predicted_binary_label = ""
    fallback_xgb_p_dead = np.nan
    fallback_xgb_p_alive = np.nan
    fallback_xgb_p_weak = np.nan
    fallback_xgb_p_strong = np.nan
    fallback_xgb_alive_threshold = np.nan

    fallback_ica = None

    # Fallback camera-correction diagnostics.
    fallback_motion_correction_used = 0
    fallback_selected_correction_sign = np.nan
    fallback_camera_motion_csv = ""
    fallback_camera_motion_median_px = np.nan
    fallback_camera_motion_p95_px = np.nan
    fallback_camera_motion_max_px = np.nan
    fallback_camera_motion_ecc_failed_n = np.nan
    fallback_outside_p95_sign_neg1 = np.nan
    fallback_outside_p95_sign_pos1 = np.nan
    fallback_masks_motion_corrected = 0
    fallback_skipped_alive_by_slight_motion = int(
        mobility_alive_by_slight_motion == 1
    )
    fallback_skipped_alive_by_any_motion = int(
        mobility_alive_by_motion_tier == 1
    )

    # New: mobility/fish-displacement evidence
    OVERLAP_THRESHOLD = 0.85
    epoch_mask_overlap_frac = np.nan
    reusable_union_mask_npy = ""
    reusable_fallback_metadata_json = ""

    # Keep the existing output-column name for compatibility, but it now
    # represents either strong motion or repeated slight motion.
    mobility_alive_by_pos_dev = int(mobility_alive_by_motion_tier)

    mobility_alive_by_overlap = 0
    mobility_alive_final = mobility_alive_by_pos_dev

    # Any accepted motion tier is already sufficient evidence of life.
    # Only NO-MOTION wells proceed to SAM/ICA/XGBoost fallback.
    if mobility_alive_by_motion_tier:
        longest_epoch_fallback_reason = (
            f"skipped_{mobility_motion_tier}"
        )
        print(
            f"[INFO] well={well}: skipping fallback because "
            f"mobility tier is {mobility_motion_tier}."
        )

    elif longest_baseline_epoch is None:
        longest_epoch_fallback_reason = "no_baseline_epoch_found"

    elif longest_baseline_epoch["duration_sec"] < args.fallback_min_epoch_sec:
        longest_epoch_fallback_reason = (
            f"longest_baseline_too_short_"
            f"{longest_baseline_epoch['duration_sec']:.3f}s"
        )

    else:
        try:
            print(
                f"[INFO] Loading SAM for epoch-only fallback on "
                f"longest baseline epoch."
            )

            # Delay heavy SAM imports until fallback is actually needed.
            from sam_utils import (
                DEFAULT_SAM_CHECKPOINT,
                load_sam,
                make_mask_generator,
                run_sam_on_rgb_hb,
                SamRunConfig,
            )

            sam = load_sam(
                checkpoint_path=args.checkpoint or DEFAULT_SAM_CHECKPOINT,
                model_type=args.model_type,
                device=args.device,
            )
            mask_generator = make_mask_generator(sam)

            cfg = SamRunConfig(
                lower_area=args.lower_area,
                upper_area=args.upper_area,
                save_overlay=False,
                save_label_image=False,
                save_individual_masks=False,
            )

            epoch_start_frame = 0
            epoch_end_frame = 899

            # Arena circle mapped into the 1048 crop for the HB mask selector.
            arena_circle = (
                res["cx"] - crop_xoff,
                res["cy"] - crop_yoff,
                res["radius"],
            )

            # Search for valid masks on the original frames. The resulting
            # masks are then translated into the same corrected coordinate
            # system used by the fallback ICA frames.
            if save_sam_debug_masks:
                debug_sam_start = debug_run_sam_on_rgb_hb_factory(
                    out_dir=out_dir,
                    well=well,
                    tag="epoch_start_forward",
                    run_sam_on_rgb_fn=run_sam_on_rgb_hb,
                    arena_circle=arena_circle,
                )

                debug_sam_end = debug_run_sam_on_rgb_hb_factory(
                    out_dir=out_dir,
                    well=well,
                    tag="epoch_end_backward",
                    run_sam_on_rgb_fn=run_sam_on_rgb_hb,
                    arena_circle=arena_circle,
                )
            else:
                # Still emit candidate_scores/survivors.csv (no heavy debug dump)
                # by threading out_dir/base_name into the 3-arg search call.
                def _hb_with_scores(tag):
                    def _run(mask_generator, rgb, cfg):
                        return run_sam_on_rgb_hb(
                            mask_generator, rgb, cfg,
                            out_dir=out_dir,
                            base_name=f"well_{well}_{tag}",
                            arena_circle=arena_circle,
                        )
                    return _run

                debug_sam_start = _hb_with_scores("epoch_start_forward")
                debug_sam_end = _hb_with_scores("epoch_end_backward")

            search_cap = cv2.VideoCapture(str(videopath))
            if not search_cap.isOpened():
                raise RuntimeError(f"Could not open: {videopath}")

            try:
                mask_a, idx_a, rgb_a, xoff_a, yoff_a = (
                    find_valid_mask_by_search(
                        videopath=str(videopath),
                        mask_generator=mask_generator,
                        run_sam_on_rgb_fn=debug_sam_start,
                        cfg=cfg,
                        start_frame=epoch_start_frame,
                        max_search=args.search_max,
                        direction="forward",
                        crop_size=args.crop_size,
                        fixed_xoff=crop_xoff,
                        fixed_yoff=crop_yoff,
                        cap=search_cap,
                    )
                )

                mask_b, idx_b, rgb_b, xoff_b, yoff_b = (
                    find_valid_mask_by_search(
                        videopath=str(videopath),
                        mask_generator=mask_generator,
                        run_sam_on_rgb_fn=debug_sam_end,
                        cfg=cfg,
                        start_frame=epoch_end_frame,
                        max_search=args.search_max,
                        direction="backward",
                        crop_size=args.crop_size,
                        fixed_xoff=crop_xoff,
                        fixed_yoff=crop_yoff,
                        cap=search_cap,
                    )
                )
            finally:
                search_cap.release()

            if mask_a is None and mask_b is None:
                longest_epoch_fallback_reason = (
                    "epoch_no_valid_masks"
                )

            elif mask_a is None or mask_b is None:
                longest_epoch_fallback_reason = (
                    "epoch_only_one_valid_mask_found"
                )

            else:
                # No-motion-correction variant: keep endpoint masks as SAM found
                # on original frames and continue with overlap/union logic.
                fallback_camera_motion_df = None
                fallback_selected_correction_sign = -1
                fallback_motion_correction_used = 0
                fallback_masks_motion_corrected = 0

                mask_a_corrected = np.asarray(mask_a, dtype=bool)
                mask_b_corrected = np.asarray(mask_b, dtype=bool)

                overlap_frac = (
                    mask_overlap_fraction_over_smaller(
                        mask_a_corrected,
                        mask_b_corrected,
                    )
                )
                epoch_mask_overlap_frac = float(overlap_frac)

                print(
                    f"[INFO] well={well} epoch mask overlap "
                    f"(intersection / smaller mask area) = "
                    f"{overlap_frac:.4f}"
                )

                # Overlap is now QC only. It no longer sets final Alive/Dead.
                # Every no-motion well with two valid masks proceeds to the
                # feature/XGBoost fallback.
                mobility_alive_by_overlap = int(
                    0.0 < overlap_frac < OVERLAP_THRESHOLD
                )

                bbox_crop, union = union_bbox_from_masks(
                    mask_a_corrected,
                    mask_b_corrected,
                    pad=args.pad,
                    dilate_px=args.dilate_px,
                    min_size=args.min_size,
                )

                if bbox_crop is None or union is None:
                    longest_epoch_fallback_reason = (
                        "epoch_union_bbox_failed"
                    )

                else:
                    if reusable_well_dir is not None:
                        union_mask_dir = reusable_well_dir
                        union_mask_npy = (
                            reusable_well_dir
                            / "fallback_union_mask.npy"
                        )
                        reusable_union_mask_npy = str(
                            union_mask_npy
                        )
                    else:
                        union_mask_dir = (
                            out_dir
                            / "longest_baseline_epoch_union_masks"
                        )
                        union_mask_npy = (
                            union_mask_dir
                            / (
                                f"well_{well}_longest_baseline_epoch_"
                                f"union_mask.npy"
                            )
                        )

                    ensure_dir(union_mask_dir)
                    np.save(
                        union_mask_npy,
                        union.astype(np.uint8),
                    )
                    epoch_union_mask_npy = str(union_mask_npy)

                    if reusable_well_dir is not None:
                        print(
                            f"[INFO] saved reusable fallback union mask: "
                            f"{union_mask_npy}"
                        )

                    # Save corrected endpoint masks for QC. These are
                    # binary masks only; no corrected video is saved.
                    cv2.imwrite(
                        str(
                            union_mask_dir
                            / (
                                f"well_{well}_epoch_start_mask_"
                                f"motion_corrected.png"
                            )
                        ),
                        mask_a_corrected.astype(np.uint8) * 255,
                    )
                    cv2.imwrite(
                        str(
                            union_mask_dir
                            / (
                                f"well_{well}_epoch_end_mask_"
                                f"motion_corrected.png"
                            )
                        ),
                        mask_b_corrected.astype(np.uint8) * 255,
                    )

                    fallback_ica = (
                        run_fallback_ica_on_crop_mask_epoch(
                            videopath=str(videopath),
                            mask_crop=union.astype(bool),
                            xoff=int(crop_xoff),
                            yoff=int(crop_yoff),
                            fps_use=float(fps),
                            frame_start=epoch_start_frame,
                            frame_end=epoch_end_frame,
                            n_keep=int(
                                args.ica_n_components
                            ),
                            out_dir=out_dir,
                            well=well,
                            ica_fun=str(args.ica_fun),
                            ica_max_iter=int(
                                args.ica_max_iter
                            ),
                            ica_tol=float(args.ica_tol),
                            ica_random_state=int(
                                args.ica_random_state
                            ),
                            camera_motion_df=(
                                fallback_camera_motion_df
                            ),
                            correction_sign=int(
                                fallback_selected_correction_sign
                            ),
                            xgb_model_path=str(
                                args.xgb_model_path
                            ),
                            fallback_model_mode=str(
                                args.fallback_model_mode
                            ),
                            xgb_alive_threshold=float(
                                args.xgb_alive_threshold
                            ),
                        )
                    )

                    # Mark fallback used only after corrected ICA
                    # finishes successfully.
                    used_longest_epoch_fallback = 1

                    epoch_ica_summary_csv = (
                        fallback_ica["ic_summary_csv"]
                    )
                    epoch_ica_time_pngs = (
                        fallback_ica["time_signal_pngs"]
                    )
                    epoch_ica_spatial_pngs = (
                        fallback_ica["spatial_map_pngs"]
                    )
                    epoch_blob_summary_png = (
                        fallback_ica["blob_summary_png"]
                    )
                    epoch_blob_trace_pngs = (
                        fallback_ica["blob_trace_pngs"]
                    )
                    fallback_fixed_235_feature_csv = (
                        fallback_ica["fixed_235_feature_csv"]
                    )
                    fallback_fixed_235_feature_names_txt = (
                        fallback_ica["fixed_235_feature_names_txt"]
                    )
                    fallback_fixed_235_feature_count = int(
                        fallback_ica["fixed_235_feature_count"]
                    )
                    fallback_xgb_no_interval_cv_feature_csv = (
                        fallback_ica["xgb_no_interval_cv_feature_csv"]
                    )
                    fallback_xgb_no_interval_cv_feature_names_txt = (
                        fallback_ica["xgb_no_interval_cv_feature_names_txt"]
                    )
                    fallback_xgb_no_interval_cv_feature_count = int(
                        fallback_ica["xgb_no_interval_cv_feature_count"]
                    )

                    fallback_xgb_model_mode = str(
                        fallback_ica["xgb_model_mode"]
                    )
                    fallback_xgb_predicted_multiclass_int = (
                        fallback_ica["xgb_predicted_multiclass_int"]
                    )
                    fallback_xgb_predicted_multiclass_label = str(
                        fallback_ica["xgb_predicted_multiclass_label"]
                    )
                    fallback_xgb_predicted_binary = int(
                        fallback_ica["xgb_predicted_binary"]
                    )
                    fallback_xgb_predicted_binary_label = str(
                        fallback_ica["xgb_predicted_binary_label"]
                    )
                    fallback_xgb_p_dead = float(
                        fallback_ica["xgb_p_dead"]
                    )
                    fallback_xgb_p_alive = float(
                        fallback_ica["xgb_p_alive"]
                    )
                    fallback_xgb_p_weak = fallback_ica["xgb_p_weak"]
                    fallback_xgb_p_strong = fallback_ica["xgb_p_strong"]
                    fallback_xgb_alive_threshold = (
                        fallback_ica["xgb_alive_threshold"]
                    )

                    # Final fallback authority:
                    # motion-positive wells never reach here;
                    # for no-motion wells, XGBoost decides Alive/Dead.
                    mobility_alive_final = int(
                        fallback_xgb_predicted_binary
                    )

                    longest_epoch_fallback_reason = (
                        "xgboost_fallback_complete"
                    )

                    if reusable_well_dir is not None:
                        fallback_meta_path = (
                            reusable_well_dir
                            / "fallback_metadata.json"
                        )
                        _save_json_safe(
                            fallback_meta_path,
                            {
                                "well": well,
                                "plate": plate_name,
                                "epoch_start_frame": int(epoch_start_frame),
                                "epoch_end_frame": int(epoch_end_frame),
                                "epoch_union_mask_npy": str(epoch_union_mask_npy),
                                "reusable_union_mask_npy": str(reusable_union_mask_npy),
                                "fallback_selected_correction_sign": int(fallback_selected_correction_sign),
                            },
                        )
                        reusable_fallback_metadata_json = str(
                            fallback_meta_path
                        )

        except Exception as e:
            used_longest_epoch_fallback = 0
            fallback_ica = None
            longest_epoch_fallback_reason = (
                f"epoch_fallback_failed: {e}"
            )

    # ---------- optional detailed CSV exports ----------
    if save_optional_qc:
        ts_csv = out_dir / f"well_{well}_inside_minus_outside_timeseries.csv"
        with open(ts_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "well", "frame_idx", "time_sec",
                "inside_norm", "outside_norm", "plate_edge_norm",
                "fitted_inside_background",
                "motion_sub",
                "motion_sub_for_t2",
                "outside_artifact",
                "edge_artifact",
                "plate_artifact",
                "mode_val",
                "baseline_mean",
                "pos_dev",
                "pos_dev_threshold",
                "state",
                "state_epoch_for_longest_baseline",
            ])
            for i in range(len(motion_sub)):
                writer.writerow([
                    well,
                    int(valid_frame_idxs[i]),
                    float(t[i]),
                    float(valid_in_norm[i]),
                    float(valid_out_norm[i]),
                    float(valid_plate_edge_norm[i]) if np.isfinite(valid_plate_edge_norm[i]) else np.nan,
                    float(fitted_inside_background[i]),
                    float(motion_sub[i]),
                    float(motion_sub_for_t2[i]) if np.isfinite(motion_sub_for_t2[i]) else np.nan,
                    int(outside_artifact[i]),
                    int(edge_artifact[i]),
                    int(plate_artifact[i]),
                    float(mode_val),
                    float(mu_base),
                    float(pos_dev[i]) if np.isfinite(pos_dev[i]) else np.nan,
                    float(POS_DEV_THRESHOLD),
                    int(state[i]),
                    int(state_epoch[i]),
                ])

        seg_csv = out_dir / f"well_{well}_inside_minus_outside_epochs.csv"
        with open(seg_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "well", "epoch_idx", "state",
                "start_idx", "end_idx",
                "start_frame", "end_frame",
                "start_time", "end_time",
                "duration_sec", "n_points"
            ])
            for j, ep in enumerate(epochs):
                writer.writerow([
                    well,
                    j,
                    state_name(ep["state"]),
                    int(ep["start_idx"]),
                    int(ep["end_idx"]),
                    int(ep["start_frame"]),
                    int(ep["end_frame"]),
                    float(ep["start_time"]),
                    float(ep["end_time"]),
                    float(ep["duration_sec"]),
                    int(ep["n_points"]),
                ])

    # ---------- append summary ----------
    if int(mobility_alive_final) == 1:
        if mobility_alive_by_strong_motion:
            mobility_alive_call = "alive_by_motion"
        elif mobility_alive_by_slight_motion:
            mobility_alive_call = "alive_by_slight_motion"
        elif fallback_xgb_predicted_binary == 1:
            mobility_alive_call = "alive_by_xgboost_fallback"
        else:
            mobility_alive_call = "alive"
    else:
        if fallback_xgb_predicted_binary == 0:
            mobility_alive_call = "dead_by_xgboost_fallback"
        else:
            mobility_alive_call = "Unknown"

    # ==========================================================
    # CLEAN PLATE SUMMARY
    # ==========================================================
    # Keep only decision-critical information plus compact QC.
    # Detailed debugging remains in per-well plots/CSVs/profiling outputs.
    summary_row = {
        # Identity
        "plate": plate_name,
        "well": well,
        "videopath": str(videopath),
        "fps": float(res_in["fps"]),

        # Compact ring/crop QC
        "circle_radius": float(res["radius"]),
        "circle_rmse": float(res["rmse"]),
        "center_crop_xoff": int(crop_xoff),
        "center_crop_yoff": int(crop_yoff),

        # Motion / edge-veto QC
        "plate_veto_fraction": (
            float(np.mean(plate_artifact))
            if len(plate_artifact) > 0
            else np.nan
        ),
        "max_pos_dev_from_baseline": float(max_pos_dev_from_baseline),
        "strong_motion_peak_threshold": float(
            STRONG_MOTION_PEAK_THRESHOLD
        ),
        "slight_motion_peak_low": float(SLIGHT_MOTION_PEAK_LOW),
        "slight_motion_peak_high": float(
            STRONG_MOTION_PEAK_THRESHOLD
        ),
        "n_motion_peaks_total_ge_0p01": int(
            len(motion_peak_indices)
        ),
        "n_strong_motion_peaks": int(n_strong_motion_peaks),
        "n_slight_motion_peaks": int(n_slight_motion_peaks),
        "mobility_alive_by_strong_motion": int(
            mobility_alive_by_strong_motion
        ),
        "mobility_alive_by_slight_motion": int(
            mobility_alive_by_slight_motion
        ),
        "mobility_motion_tier": mobility_motion_tier,

        # Overlap is QC only
        "epoch_mask_overlap_frac": (
            float(epoch_mask_overlap_frac)
            if np.isfinite(epoch_mask_overlap_frac)
            else np.nan
        ),
        "mobility_alive_by_overlap": int(
            mobility_alive_by_overlap
        ),

        # Final decision
        "mobility_alive_final": int(mobility_alive_final),
        "mobility_alive_call": mobility_alive_call,

        # Fallback status
        "used_longest_epoch_fallback": int(
            used_longest_epoch_fallback
        ),
        "longest_epoch_fallback_reason": str(
            longest_epoch_fallback_reason
        ),
        "fallback_skipped_alive_by_any_motion": int(
            fallback_skipped_alive_by_any_motion
        ),

        # Feature/QC outputs
        "epoch_ica_summary_csv": str(epoch_ica_summary_csv),
        "fallback_fixed_235_feature_csv": str(
            fallback_fixed_235_feature_csv
        ),
        "fallback_xgb_no_interval_cv_feature_csv": str(
            fallback_xgb_no_interval_cv_feature_csv
        ),

        # XGBoost prediction
        "fallback_xgb_model_mode": str(
            fallback_xgb_model_mode
        ),
        "fallback_xgb_predicted_multiclass_int": (
            fallback_xgb_predicted_multiclass_int
        ),
        "fallback_xgb_predicted_multiclass_label": str(
            fallback_xgb_predicted_multiclass_label
        ),
        "fallback_xgb_predicted_binary": (
            fallback_xgb_predicted_binary
        ),
        "fallback_xgb_predicted_binary_label": str(
            fallback_xgb_predicted_binary_label
        ),

        # XGBoost probabilities
        "fallback_xgb_p_dead": fallback_xgb_p_dead,
        "fallback_xgb_p_alive": fallback_xgb_p_alive,
        "fallback_xgb_p_weak": fallback_xgb_p_weak,
        "fallback_xgb_p_strong": fallback_xgb_p_strong,

        # ICA QC
        "epoch_ica_converged": (
            int(fallback_ica["ica_converged"])
            if fallback_ica is not None
            else np.nan
        ),
        "epoch_ica_n_iter": (
            fallback_ica["ica_n_iter"]
            if fallback_ica is not None
            else np.nan
        ),
    }
    save_summary_row(out_dir / "plate_summary.csv", summary_row)
  
    

    print(f"[DONE] {well}")
    print(f"  videopath   : {videopath}")
    print(f"  circle      : ({res['cx']:.2f}, {res['cy']:.2f}), r={res['radius']:.2f}, rmse={res['rmse']:.3f}")
    print(f"  1048 crop   : xoff={crop_xoff}, yoff={crop_yoff}")
    print(f"  mode        : {mode_val:.6f}")
    print(f"  n epochs    : {len(epochs)} (below: {len(below_epochs)}, baseline: {len(baseline_epochs)}, above: {len(above_epochs)})")

    if longest_baseline_epoch is not None:
        print(
            f"  longest baseline: {longest_baseline_epoch['duration_sec']:.3f}s "
            f"(frames {longest_baseline_epoch['start_frame']} to {longest_baseline_epoch['end_frame']})"
        )

    print(f"  epoch fallback used: {used_longest_epoch_fallback}")
    if longest_epoch_fallback_reason:
        print(f"  epoch fallback reason: {longest_epoch_fallback_reason}")

    print(f"  output dir  : {out_dir}")


def debug_run_sam_on_rgb_hb_factory(out_dir, well, tag, run_sam_on_rgb_fn, arena_circle=None):
    """
    Keep current HB SAM behavior, but save:
      1. all raw SAM masks before filtering
      2. final HB-filtered masks after run_sam_on_rgb_hb
    """

    call_counter = {"n": 0}

    def debug_run_sam_on_rgb_hb(mask_generator, rgb, cfg):
        call_counter["n"] += 1

        debug_dir = (
            Path(out_dir)
            / "debug_masks_area_filtered"
            / f"well_{well}_{tag}_call{call_counter['n']:03d}"
        )
        debug_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = debug_dir / "raw_masks"
        final_dir = debug_dir / "final_hb_filtered_masks"

        raw_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # Save the RGB crop/frame sent to SAM
        # -------------------------------------------------
        cv2.imwrite(
            str(debug_dir / "crop_rgb.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )

        # -------------------------------------------------
        # 1) Save ALL RAW SAM masks before HB filtering
        # -------------------------------------------------
        raw_masks = mask_generator.generate(rgb)

        raw_rows = []

        for i, m in enumerate(raw_masks, start=1):
            seg = m["segmentation"].astype(bool)
            area = int(m.get("area", int(seg.sum())))

            touches_border = int(
                seg[0, :].any()
                or seg[-1, :].any()
                or seg[:, 0].any()
                or seg[:, -1].any()
            )

            passes_area = int(cfg.lower_area <= area <= cfg.upper_area)

            raw_rows.append({
                "mask_id": i,
                "area": area,
                "passes_area": passes_area,
                "touches_border": touches_border,
                "predicted_iou": m.get("predicted_iou", np.nan),
                "stability_score": m.get("stability_score", np.nan),
            })

            # binary raw mask
            cv2.imwrite(
                str(
                    raw_dir
                    / f"raw_mask_{i:03d}_area{area}_pass{passes_area}_border{touches_border}.png"
                ),
                seg.astype(np.uint8) * 255,
            )

            # raw overlay
            overlay = rgb.copy()
            overlay[seg] = [0, 255, 0]

            cv2.imwrite(
                str(
                    raw_dir
                    / f"raw_overlay_{i:03d}_area{area}_pass{passes_area}_border{touches_border}.png"
                ),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        raw_csv_path = debug_dir / "raw_mask_summary.csv"
        with open(raw_csv_path, "w", newline="") as f:
            fieldnames = [
                "mask_id",
                "area",
                "passes_area",
                "touches_border",
                "predicted_iou",
                "stability_score",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_rows)

        # -------------------------------------------------
        # 2) Run the REAL current HB SAM function
        # -------------------------------------------------
        hb_result = run_sam_on_rgb_fn(
            mask_generator, rgb, cfg,
            out_dir=out_dir,
            base_name=f"well_{well}_{tag}_call{call_counter['n']:03d}",
            arena_circle=arena_circle,
        )

        final_masks = hb_result.get("masks", [])

        final_rows = []

        for i, m in enumerate(final_masks, start=1):
            seg = m["segmentation"].astype(bool)
            area = int(m.get("area", int(seg.sum())))

            touches_border = int(
                seg[0, :].any()
                or seg[-1, :].any()
                or seg[:, 0].any()
                or seg[:, -1].any()
            )

            final_rows.append({
                "final_mask_id": i,
                "area": area,
                "touches_border": touches_border,
                "predicted_iou": m.get("predicted_iou", np.nan),
                "stability_score": m.get("stability_score", np.nan),
            })

            # final binary mask
            cv2.imwrite(
                str(final_dir / f"final_mask_{i:03d}_area{area}_border{touches_border}.png"),
                seg.astype(np.uint8) * 255,
            )

            # final overlay
            overlay = rgb.copy()
            overlay[seg] = [255, 255, 0]  # yellow final mask

            cv2.imwrite(
                str(final_dir / f"final_overlay_{i:03d}_area{area}_border{touches_border}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        final_csv_path = debug_dir / "final_hb_filtered_mask_summary.csv"
        with open(final_csv_path, "w", newline="") as f:
            fieldnames = [
                "final_mask_id",
                "area",
                "touches_border",
                "predicted_iou",
                "stability_score",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)

        # -------------------------------------------------
        # Optional combined quick text summary
        # -------------------------------------------------
        summary_txt = debug_dir / "debug_summary.txt"
        with open(summary_txt, "w") as f:
            f.write(f"well: {well}\n")
            f.write(f"tag: {tag}\n")
            f.write(f"call: {call_counter['n']}\n")
            f.write(f"lower_area: {cfg.lower_area}\n")
            f.write(f"upper_area: {cfg.upper_area}\n")
            f.write(f"n_raw_masks: {len(raw_masks)}\n")
            f.write(f"n_final_hb_filtered_masks: {len(final_masks)}\n")
            f.write(f"raw_csv: {raw_csv_path}\n")
            f.write(f"final_csv: {final_csv_path}\n")

        print(
            f"[SAM DEBUG] Saved raw={len(raw_masks)} and "
            f"final_hb_filtered={len(final_masks)} masks to: {debug_dir}"
        )

        # IMPORTANT: return the actual HB result
        return hb_result

    return debug_run_sam_on_rgb_hb
if __name__ == "__main__":
    main()