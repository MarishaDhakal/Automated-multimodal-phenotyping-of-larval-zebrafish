#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import median_filter

from sam_utils import (
    load_sam,
    make_mask_generator,
    run_sam_on_rgb,
    save_results,
    SamRunConfig,
    center_crop_rgb,
)

from fluro_utils import (
    estimate_circle_crop_params_annulus,
    apply_circle_crop,
    save_ring_debug_image,
)


# ============================================================
# DEFAULTS
# ============================================================
VIDEO_NAME_PATTERN_DEFAULT = "well_{well}.mp4"
TARGET_FRAMES_DEFAULT = "0,7,14,22,29"

CROP_SIZE_DEFAULT = 0
DEVICE_DEFAULT = "cpu"

LOWER_AREA_DEFAULT = 0
UPPER_AREA_DEFAULT = 0

BIN_OFFSET_DEFAULT = 8.0
BIN_MORPH_RADIUS_DEFAULT = 6
INNER_MORPH_RADIUS_DEFAULT = 8
BIN_BLUR_SIGMA_DEFAULT = 2.0
BIN_INSET_DEFAULT = 40
RING_RADIUS_PAD_DEFAULT = 15.0


# ============================================================
# HELPERS
# ============================================================
def read_mp4_frame_rgb(mp4_path: Path, frame_idx: int) -> tuple[np.ndarray, dict]:
    """
    Read one MP4 frame as RGB.

    Minimal new behavior:
    - If original frame is 1536x1536:
        return as-is, no median filter.
    - If original frame is 3072x3072:
        downscale to 1536x1536, then apply 2D median filter.
    """
    cap = cv2.VideoCapture(str(mp4_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {mp4_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        raise RuntimeError(f"Failed reading frame {frame_idx} from {mp4_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    H0, W0 = frame_rgb.shape[:2]

    info = {
        "original_w": int(W0),
        "original_h": int(H0),
        "working_w": int(W0),
        "working_h": int(H0),
        "was_downscaled": False,
        "median_filter_applied": False,
    }

    if H0 == 1536 and W0 == 1536:
        return frame_rgb, info

    elif H0 == 3072 and W0 == 3072:
        frame_rgb = cv2.resize(
            frame_rgb,
            (1536, 1536),
            interpolation=cv2.INTER_AREA,
        )

        # frame_rgb = median_filter(
        #     frame_rgb,
        #     size=(5, 5, 1)
        # ).astype(np.uint8)

        info["working_w"] = 1536
        info["working_h"] = 1536
        info["was_downscaled"] = True
        info["median_filter_applied"] = False

        return frame_rgb, info

    else:
        raise ValueError(
            f"Unsupported video size {W0}x{H0}. "
            "Expected 1536x1536 or 3072x3072."
        )


def choose_video_settings_from_frame(
    frame_rgb: np.ndarray,
    *,
    crop_size_arg: int | None = None,
    lower_area_arg: int | None = None,
    upper_area_arg: int | None = None,
):
    """
    After read_mp4_frame_rgb(), all accepted frames are processed
    at 1536x1536 working resolution.

    Settings:
        crop_size = 1048
        lower_area = 8000
        upper_area = 35000
        annulus_radius_override = 500
    """
    H, W = frame_rgb.shape[:2]

    if H != 1536 or W != 1536:
        raise ValueError(
            f"Working frame must be 1536x1536, got {W}x{H}."
        )

    profile = "1536_working"

    auto_crop_size = 1048
    auto_lower_area = 8000
    auto_upper_area = 35000
    annulus_radius_override = 500.0

    crop_size = auto_crop_size if crop_size_arg is None else crop_size_arg
    lower_area = auto_lower_area if lower_area_arg is None else lower_area_arg
    upper_area = auto_upper_area if upper_area_arg is None else upper_area_arg

    return {
        "profile": profile,
        "video_w": int(W),
        "video_h": int(H),
        "crop_size": int(crop_size),
        "lower_area": int(lower_area),
        "upper_area": int(upper_area),
        "annulus_radius_override": annulus_radius_override,
    }


def choose_largest_mask(masks):
    if not masks:
        return None
    return max(masks, key=lambda m: int(m["segmentation"].sum()))


def overlay_best_mask(rgb: np.ndarray, roi: np.ndarray, fill_alpha: float = 0.35) -> np.ndarray:
    vis = rgb.copy()
    roi = roi.astype(bool)

    if roi.any():
        fill = np.zeros_like(vis)
        fill[roi] = (0, 255, 0)
        vis = (vis * (1 - fill_alpha) + fill * fill_alpha).astype(np.uint8)

        edges = cv2.Canny((roi.astype(np.uint8) * 255), 50, 150)
        vis[edges > 0] = (255, 0, 0)

    return vis


def compute_metrics_from_roi(rgb: np.ndarray, roi: np.ndarray) -> dict:
    G = rgb[..., 1].astype(np.float32)
    vals = G[roi.astype(bool)]

    if vals.size == 0:
        return {
            "mean_green": np.nan,
            "top10_green": np.nan,
            "top25_green": np.nan,
            "p90_green": np.nan,
            "p95_green": np.nan,
            "p99_green": np.nan,
            "max_green": np.nan,
        }

    mean_green = float(vals.mean())

    k10 = max(1, int(0.10 * vals.size))
    top10_green = float(np.partition(vals, -k10)[-k10:].mean())

    k25 = max(1, int(0.25 * vals.size))
    top25_green = float(np.partition(vals, -k25)[-k25:].mean())

    return {
        "mean_green": mean_green,
        "top10_green": top10_green,
        "top25_green": top25_green,
        "p90_green": float(np.percentile(vals, 90)),
        "p95_green": float(np.percentile(vals, 95)),
        "p99_green": float(np.percentile(vals, 99)),
        "max_green": float(vals.max()),
    }


def parse_frames(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def valid_well_name(well: str) -> str:
    well = well.upper().strip()

    if len(well) < 2:
        raise ValueError(f"Invalid well name: {well}")

    row = well[0]
    col = well[1:]

    if row not in "ABCDEFGH":
        raise ValueError(f"Invalid row in well name: {well}")

    if not col.isdigit():
        raise ValueError(f"Invalid column in well name: {well}")

    col_i = int(col)
    if col_i < 1 or col_i > 12:
        raise ValueError(f"Invalid column in well name: {well}")

    return f"{row}{col_i}"


def compute_slopes(df_in: pd.DataFrame) -> pd.DataFrame:
    ok = df_in[df_in["note"].fillna("") == ""].copy()

    if ok.empty:
        return pd.DataFrame([{
            "well": df_in["well"].iloc[0] if len(df_in) else "",
            "n_points": 0,
            "slope_mean_green": np.nan,
            "r2_mean_green": np.nan,
            "slope_top10_green": np.nan,
            "r2_top10_green": np.nan,
            "slope_top25_green": np.nan,
            "r2_top25_green": np.nan,
        }])

    rows = []

    for well, g in ok.groupby("well"):
        g = g.sort_values("frame")
        x = g["frame"].to_numpy(dtype=float)

        def fit_line(y):
            y = np.asarray(y, dtype=float)
            m = np.isfinite(x) & np.isfinite(y)

            if m.sum() < 2:
                return np.nan, np.nan

            xf = x[m]
            yf = y[m]

            a, b = np.polyfit(xf, yf, 1)
            yhat = a * xf + b

            ss_res = np.sum((yf - yhat) ** 2)
            ss_tot = np.sum((yf - np.mean(yf)) ** 2)

            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            return float(a), float(r2)

        slope_mean, r2_mean = fit_line(g["mean_green"].to_numpy())
        slope_top10, r2_top10 = fit_line(g["top10_green"].to_numpy())
        slope_top25, r2_top25 = fit_line(g["top25_green"].to_numpy())

        rows.append({
            "well": well,
            "n_points": int(len(g)),
            "slope_mean_green": slope_mean,
            "r2_mean_green": r2_mean,
            "slope_top10_green": slope_top10,
            "r2_top10_green": r2_top10,
            "slope_top25_green": slope_top25,
            "r2_top25_green": r2_top25,
        })

    return pd.DataFrame(rows)


# ============================================================
# CORE RUN: ONE WELL
# ============================================================
def run_one_well(
    plate_dir: Path,
    out_root: Path,
    well: str,
    video_name_pattern: str,
    target_frames,
    crop_size: int | None,
    device: str,
    lower_area: int | None,
    upper_area: int | None,
    bin_offset: float,
    bin_inset: int,
    bin_blur_sigma: float,
    bin_morph_radius: int,
    inner_morph_radius: int,
    reusable_parent_dir: str = "",
):
    plate_dir = plate_dir.resolve()

    if not plate_dir.exists():
        raise FileNotFoundError(f"Plate directory not found: {plate_dir}")

    well = valid_well_name(well)
    plate_name = plate_dir.name
    base_name = f"well_{well}"

    plate_out_dir = (out_root / f"{plate_name}_per_well_fluorescence").resolve()
    plate_out_dir.mkdir(parents=True, exist_ok=True)

    out_dir = plate_out_dir

    reusable_well_dir = None
    if str(reusable_parent_dir).strip() != "":
        reusable_well_dir = Path(reusable_parent_dir) / plate_out_dir.name / base_name
        reusable_well_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] reusable artifact dir: {reusable_well_dir}")
    ring_params_path = (
        reusable_well_dir / "ring_params.json"
        if reusable_well_dir is not None else None
    )

    mp4_path = plate_dir / video_name_pattern.format(well=well)

    if not mp4_path.exists():
        raise FileNotFoundError(f"Missing video for {well}: {mp4_path}")

    print("=" * 80)
    print(f"Plate: {plate_name}")
    print(f"Well: {well}")
    print(f"Video: {mp4_path}")
    print(f"Output: {out_dir}")
    print("=" * 80)

    # ------------------------------------------------------------
    # Read first frame once and auto-select settings
    # ------------------------------------------------------------
    fi0 = int(target_frames[0]) if target_frames else 0
    frame0_rgb, norm_info0 = read_mp4_frame_rgb(mp4_path, fi0)

    settings = choose_video_settings_from_frame(
        frame0_rgb,
        crop_size_arg=crop_size,
        lower_area_arg=lower_area,
        upper_area_arg=upper_area,
    )

    crop_size = settings["crop_size"]
    lower_area = settings["lower_area"]
    upper_area = settings["upper_area"]
    annulus_radius_override = settings["annulus_radius_override"]

    # Since all frames are now processed as 1536 working resolution
    annulus_min_area = 100000

    print(
        f"[SETTINGS] original_video={norm_info0['original_w']}x{norm_info0['original_h']} | "
        f"working_video={settings['video_w']}x{settings['video_h']} | "
        f"profile={settings['profile']} | "
        f"was_downscaled={norm_info0['was_downscaled']} | "
        f"median_filter_applied={norm_info0['median_filter_applied']} | "
        f"crop_size={crop_size} | "
        f"lower_area={lower_area} | "
        f"upper_area={upper_area} | "
        f"annulus_radius_override={annulus_radius_override} | "
        f"annulus_min_area={annulus_min_area}"
    )

    sam = load_sam(device=device)
    mask_gen = make_mask_generator(sam)

    sam_cfg = SamRunConfig(
        lower_area=int(lower_area),
        upper_area=int(upper_area),
        save_overlay=True,
        save_label_image=False,
        save_individual_masks=True,
    )

    rows_out = []
    radius_pad = float(RING_RADIUS_PAD_DEFAULT)

    circle_params = None
    circle_note = ""
    reused_ring_params = False

    # ------------------------------------------------------------
    # Reuse/estimate stable circle crop once per well
    # ------------------------------------------------------------
    if crop_size is not None and crop_size > 0:
        try:
            if ring_params_path is not None and ring_params_path.exists():
                ring_cache = json.loads(ring_params_path.read_text())
                cached_h = int(ring_cache.get("frame_h", frame0_rgb.shape[0]))
                cached_w = int(ring_cache.get("frame_w", frame0_rgb.shape[1]))
                cx_c = float(ring_cache["circle_cx"])
                cy_c = float(ring_cache["circle_cy"])
                r_c = float(ring_cache["circle_radius"])
                x0_c = int(ring_cache.get("x0", 0))
                y0_c = int(ring_cache.get("y0", 0))
                crop_c = int(ring_cache.get("crop_size", crop_size))

                h0, w0 = frame0_rgb.shape[:2]
                if (
                    cached_h == int(h0)
                    and cached_w == int(w0)
                    and np.isfinite(cx_c)
                    and np.isfinite(cy_c)
                    and np.isfinite(r_c)
                    and r_c > 0
                ):
                    circle_params = type("CircleParams", (), {
                        "x0": x0_c,
                        "y0": y0_c,
                        "cx": cx_c,
                        "cy": cy_c,
                        "r": r_c,
                        "crop_size": crop_c,
                    })()
                    reused_ring_params = True
                    print(f"[INFO] reused ring params from: {ring_params_path}")

            if circle_params is None:
                circle_params, binmask0, info0 = estimate_circle_crop_params_annulus(
                    frame0_rgb,
                    crop_size=int(crop_size),
                    radius_override=annulus_radius_override,
                    annulus_min_area=annulus_min_area,
                )

                if ring_params_path is not None and not reused_ring_params:
                    ring_payload = {
                        "well": base_name,
                        "plate": plate_out_dir.name,
                        "frame_h": int(frame0_rgb.shape[0]),
                        "frame_w": int(frame0_rgb.shape[1]),
                        "circle_cx": float(circle_params.cx),
                        "circle_cy": float(circle_params.cy),
                        "circle_radius": float(circle_params.r),
                        "x0": int(circle_params.x0),
                        "y0": int(circle_params.y0),
                        "crop_size": int(circle_params.crop_size),
                        "cx_crop": float(circle_params.cx) - int(circle_params.x0),
                        "cy_crop": float(circle_params.cy) - int(circle_params.y0),
                        "radius_pad": radius_pad,
                    }
                    ring_params_path.write_text(json.dumps(ring_payload, indent=2))
                    print(f"[INFO] saved ring params: {ring_params_path}")

            save_ring_debug_image(
                frame_rgb=frame0_rgb,
                circle_params=circle_params,
                out_dir=out_dir,
                base_name=base_name,
            )

            print(
                f"[OK] Circle crop initialized: "
                f"cx={circle_params.cx:.1f}, "
                f"cy={circle_params.cy:.1f}, "
                f"r={circle_params.r:.1f}"
            )

        except Exception as e:
            circle_params = None
            circle_note = f"circle_crop_init_failed:{type(e).__name__}"
            print(f"[WARN] {base_name}: circle crop failed. Falling back to center crop.")

    # ------------------------------------------------------------
    # Process selected frames
    # ------------------------------------------------------------
    for fi in target_frames:
        try:
            frame_rgb, norm_info = read_mp4_frame_rgb(mp4_path, fi)
            circle_mask = None

            crop_method = "none"
            circle_cx = np.nan
            circle_cy = np.nan
            circle_r = np.nan

            if crop_size is not None and crop_size > 0:
                if circle_params is not None:
                    crop_rgb, x0, y0 = apply_circle_crop(frame_rgb, circle_params)
                    crop_rgb = crop_rgb.copy()
                    crop_method = "circle_crop"

                    # --------------------------------------
                    # Apply circular mask
                    # --------------------------------------
                    h, w = crop_rgb.shape[:2]

                    Y, X = np.ogrid[:h, :w]

                    # convert global center to local crop coordinates
                    cx_local = circle_params.cx - x0
                    cy_local = circle_params.cy - y0
                    mask_radius = float(circle_params.r) + radius_pad

                    dist2 = (X - cx_local) ** 2 + (Y - cy_local) ** 2
                    circle_mask = dist2 <= (mask_radius) ** 2

                    # mask outside circle
                    crop_rgb[~circle_mask] = 0

                    circle_cx = float(circle_params.cx)
                    circle_cy = float(circle_params.cy)
                    circle_r = float(mask_radius)

                else:
                    crop_rgb, x0, y0 = center_crop_rgb(frame_rgb, crop_size=int(crop_size))
                    crop_method = "center_crop"

            else:
                crop_rgb = frame_rgb
                x0, y0 = 0, 0
                crop_method = "none"

            result = run_sam_on_rgb(
                mask_generator=mask_gen,
                rgb=crop_rgb,
                cfg=sam_cfg,
                out_dir=out_dir,
                base_name=f"{base_name}_f{fi:04d}",
                circle_mask=circle_mask if circle_params is not None else None,
                select_by="area",
                #select_by="fishiness",
            )

            masks_f = result.get("masks", [])
            best = choose_largest_mask(masks_f)

            if best is None:
                print(f"{base_name} f{fi:04d}: NO MASK")

                save_results(
                    out_dir=out_dir,
                    base_name=base_name,
                    frame_idx=int(fi),
                    rgb=crop_rgb,
                    result={"masks": []},
                    cfg=sam_cfg,
                )

                rows_out.append({
                    "well": well,
                    "video": str(mp4_path),
                    "frame": int(fi),
                    "settings_profile": settings["profile"],
                    "original_video_w": norm_info["original_w"],
                    "original_video_h": norm_info["original_h"],
                    "video_w": settings["video_w"],
                    "video_h": settings["video_h"],
                    "was_downscaled": norm_info["was_downscaled"],
                    "median_filter_applied": norm_info["median_filter_applied"],
                    "crop_size": crop_size if crop_size else "",
                    "lower_area": int(lower_area),
                    "upper_area": int(upper_area),
                    "annulus_radius_override": (
                        np.nan if annulus_radius_override is None else float(annulus_radius_override)
                    ),
                    "crop_method": crop_method,
                    "crop_x0": int(x0),
                    "crop_y0": int(y0),
                    "circle_cx": circle_cx,
                    "circle_cy": circle_cy,
                    "circle_r": circle_r,
                    "mask_area": 0,
                    "mean_green": np.nan,
                    "top10_green": np.nan,
                    "top25_green": np.nan,
                    "p90_green": np.nan,
                    "p95_green": np.nan,
                    "p99_green": np.nan,
                    "max_green": np.nan,
                    "predicted_iou": np.nan,
                    "stability_score": np.nan,
                    "note": "no_mask" if circle_note == "" else f"no_mask|{circle_note}",
                })

                continue

            roi = best["segmentation"].astype(bool)
            area = int(roi.sum())
            mets = compute_metrics_from_roi(crop_rgb, roi)

            ov_best = overlay_best_mask(crop_rgb, roi, fill_alpha=0.35)

            save_results(
                out_dir=out_dir,
                base_name=base_name,
                frame_idx=int(fi),
                rgb=crop_rgb,
                result={"masks": [best], "overlay": ov_best},
                cfg=sam_cfg,
            )

            print(
                f"{base_name} f{fi:04d}: "
                f"mean={mets['mean_green']:.2f}, "
                f"top10={mets['top10_green']:.2f}, "
                f"top25={mets['top25_green']:.2f}, "
                f"p99={mets['p99_green']:.2f}, "
                f"area={area}, "
                f"crop={crop_method}, "
                f"downscaled={norm_info['was_downscaled']}"
            )

            rows_out.append({
                "well": well,
                "video": str(mp4_path),
                "frame": int(fi),
                "settings_profile": settings["profile"],
                "original_video_w": norm_info["original_w"],
                "original_video_h": norm_info["original_h"],
                "video_w": settings["video_w"],
                "video_h": settings["video_h"],
                "was_downscaled": norm_info["was_downscaled"],
                "median_filter_applied": norm_info["median_filter_applied"],
                "crop_size": crop_size if crop_size else "",
                "lower_area": int(lower_area),
                "upper_area": int(upper_area),
                "annulus_radius_override": (
                    np.nan if annulus_radius_override is None else float(annulus_radius_override)
                ),
                "crop_method": crop_method,
                "crop_x0": int(x0),
                "crop_y0": int(y0),
                "circle_cx": circle_cx,
                "circle_cy": circle_cy,
                "circle_r": circle_r,
                "mask_area": int(area),
                **mets,
                "predicted_iou": float(best.get("predicted_iou", np.nan)),
                "stability_score": float(best.get("stability_score", np.nan)),
                "note": circle_note,
            })

        except Exception as e:
            print(f"[ERROR] {base_name} f{fi:04d}: {e}")

            rows_out.append({
                "well": well,
                "video": str(mp4_path),
                "frame": int(fi),
                "settings_profile": settings["profile"],
                "original_video_w": np.nan,
                "original_video_h": np.nan,
                "video_w": settings["video_w"],
                "video_h": settings["video_h"],
                "was_downscaled": np.nan,
                "median_filter_applied": np.nan,
                "crop_size": crop_size if crop_size else "",
                "lower_area": int(lower_area),
                "upper_area": int(upper_area),
                "annulus_radius_override": (
                    np.nan if annulus_radius_override is None else float(annulus_radius_override)
                ),
                "crop_method": "error",
                "crop_x0": np.nan,
                "crop_y0": np.nan,
                "circle_cx": np.nan,
                "circle_cy": np.nan,
                "circle_r": np.nan,
                "mask_area": 0,
                "mean_green": np.nan,
                "top10_green": np.nan,
                "top25_green": np.nan,
                "p90_green": np.nan,
                "p95_green": np.nan,
                "p99_green": np.nan,
                "max_green": np.nan,
                "predicted_iou": np.nan,
                "stability_score": np.nan,
                "note": f"error:{type(e).__name__}",
            })

    # ------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------
    df = pd.DataFrame(rows_out)

    df_slopes = compute_slopes(df)
    df = df.merge(df_slopes, on="well", how="left")

    csv_path = out_dir / f"{base_name}_gcamp_metrics_5frames_with_slopes.csv"
    slope_csv_path = out_dir / f"{base_name}_gcamp_slopes.csv"

    df.to_csv(csv_path, index=False)
    df_slopes.to_csv(slope_csv_path, index=False)

    print("\nSaved:")
    print(csv_path)
    print(slope_csv_path)

    return 0


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Run fluorescence/SAM metrics for ONE well."
    )

    ap.add_argument("--plate_dir", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--well", required=True, help="Example: A1, B12, H6")

    ap.add_argument("--video_name_pattern", default=VIDEO_NAME_PATTERN_DEFAULT)
    ap.add_argument("--target_frames", default=TARGET_FRAMES_DEFAULT)
    ap.add_argument("--crop_size", type=int, default=CROP_SIZE_DEFAULT)
    ap.add_argument("--device", default=DEVICE_DEFAULT)

    ap.add_argument("--lower_area", type=int, default=LOWER_AREA_DEFAULT)
    ap.add_argument("--upper_area", type=int, default=UPPER_AREA_DEFAULT)

    ap.add_argument("--bin_offset", type=float, default=BIN_OFFSET_DEFAULT)
    ap.add_argument("--bin_inset", type=int, default=BIN_INSET_DEFAULT)
    ap.add_argument("--bin_blur_sigma", type=float, default=BIN_BLUR_SIGMA_DEFAULT)
    ap.add_argument("--bin_morph_radius", type=int, default=BIN_MORPH_RADIUS_DEFAULT)
    ap.add_argument("--inner_morph_radius", type=int, default=INNER_MORPH_RADIUS_DEFAULT)
    ap.add_argument(
        "--reusable_parent_dir",
        type=str,
        default="",
        help="Optional parent folder for reusable artifacts like ring_params.json",
    )

    args = ap.parse_args()

    target_frames = parse_frames(args.target_frames)

    crop_size = None if args.crop_size == 0 else args.crop_size
    lower_area = None if args.lower_area == 0 else args.lower_area
    upper_area = None if args.upper_area == 0 else args.upper_area

    return run_one_well(
        plate_dir=args.plate_dir,
        out_root=args.out_root,
        well=args.well,
        video_name_pattern=args.video_name_pattern,
        target_frames=target_frames,
        crop_size=crop_size,
        device=args.device,
        lower_area=lower_area,
        upper_area=upper_area,
        bin_offset=args.bin_offset,
        bin_inset=args.bin_inset,
        bin_blur_sigma=args.bin_blur_sigma,
        bin_morph_radius=args.bin_morph_radius,
        inner_morph_radius=args.inner_morph_radius,
        reusable_parent_dir=args.reusable_parent_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())