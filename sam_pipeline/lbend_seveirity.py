from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.graph import route_through_array

from sam_utils import (
    DEFAULT_SAM_CHECKPOINT,
    load_sam,
    make_mask_generator,
    SamRunConfig,
    read_morphology_tif_rgb_3072,
    run_sam_on_rgb,
    save_results,
    center_crop_ring_detection_rgb,
)

from skeletonize_utils_old import (
    SkeletonOutput,
    should_skeletonize_filtered,
    skeletonize_mask,
    save_skeleton_outputs,
    fit_midline_and_severity,
    draw_and_save_poly_midline,
    save_skeleton_on_binary_figure,
    save_polyfit_debug_figure,
    detect_ring_auto_prefer_annulus,
    center_crop_origin_from_detected_center
)

CHECKPOINT_DEFAULT = DEFAULT_SAM_CHECKPOINT
MODEL_TYPE_DEFAULT = "vit_h"
DEVICE_DEFAULT = "cpu"
def apply_arena_only_to_crop(
    crop_rgb,
    cx_global,
    cy_global,
    radius,
    x0,
    y0,
    radius_pad=30,
    outside_value=0,
):
    h, w = crop_rgb.shape[:2]

    cx_crop = cx_global - x0
    cy_crop = cy_global - y0
    r = radius + radius_pad

    yy, xx = np.ogrid[:h, :w]
    arena_mask = (xx - cx_crop) ** 2 + (yy - cy_crop) ** 2 <= r ** 2

    crop_arena = crop_rgb.copy()
    crop_arena[~arena_mask] = outside_value

    return crop_arena, arena_mask


def keep_longest_skeleton_path(sk: np.ndarray) -> np.ndarray:
    """
    Keep only the longest endpoint-to-endpoint path of a skeleton.
    This removes side branches before midline fitting.
    """
    sk = sk.astype(bool)

    ys, xs = np.where(sk)
    if len(xs) < 2:
        return sk

    # Count 8-connected neighbors
    kernel = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1],
    ], dtype=np.uint8)

    neigh = ndimage.convolve(sk.astype(np.uint8), kernel, mode="constant", cval=0)

    # endpoint = skeleton pixel with exactly one neighbor
    endpoints_mask = sk & (neigh == 11)
    ey, ex = np.where(endpoints_mask)

    if len(ex) >= 2:
        endpoints = np.column_stack([ey, ex])
        D = cdist(endpoints, endpoints)
        i, j = np.unravel_index(np.argmax(D), D.shape)
        start = tuple(endpoints[i])
        end = tuple(endpoints[j])
    else:
        # fallback: use farthest pair of skeleton pixels
        pts = np.column_stack([ys, xs])
        D = cdist(pts, pts)
        i, j = np.unravel_index(np.argmax(D), D.shape)
        start = tuple(pts[i])
        end = tuple(pts[j])

    # Path should prefer skeleton pixels strongly
    cost = np.where(sk, 1.0, 1e6)

    path, _ = route_through_array(
        cost,
        start,
        end,
        fully_connected=True,
    )

    sk_longest = np.zeros_like(sk, dtype=bool)
    for y, x in path:
        sk_longest[y, x] = True

    return sk_longest

def trim_skeleton_ends(sk: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
    """
    Remove head/tail portions of skeleton based on geodesic ordering.
    """
    sk = sk.astype(bool)

    ys, xs = np.where(sk)
    if len(xs) < 10:
        return sk

    pts = np.column_stack((ys, xs))

    # --- find endpoints (same logic) ---
    kernel = np.array([
        [1,1,1],
        [1,10,1],
        [1,1,1]
    ], dtype=np.uint8)

    neigh = ndimage.convolve(sk.astype(np.uint8), kernel, mode="constant", cval=0)
    endpoints = np.column_stack(np.where(sk & (neigh == 11)))

    if len(endpoints) < 2:
        return sk

    # pick farthest endpoints
    D = cdist(endpoints, endpoints)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    start = tuple(endpoints[i])
    end = tuple(endpoints[j])

    # trace ordered path
    cost = np.where(sk, 1.0, 1e6)
    path, _ = route_through_array(cost, start, end, fully_connected=True)

    path = np.array(path)

    # --- trim ---
    n = len(path)
    k = int(trim_frac * n)

    if n < 20 or k < 1:
        return sk

    path_trim = path[k : n - k]

    # rebuild mask
    sk_trim = np.zeros_like(sk, dtype=bool)
    sk_trim[path_trim[:, 0], path_trim[:, 1]] = True

    return sk_trim

def process_one_well(
    tif_path: Path,
    output_root: Path,
    checkpoint: str,
    model_type: str,
    device: str,
    crop_size: int,
    lower_area: int,
    upper_area: int,
    reusable_parent_dir: str = "",
    min_ring_radius: float = 800.0,
) -> None:

    if not np.isfinite(min_ring_radius) or min_ring_radius <= 0:
        raise ValueError("min_ring_radius must be positive and finite")

    if not tif_path.exists():
        raise RuntimeError(f"TIF file does not exist: {tif_path}")

    output_root.mkdir(parents=True, exist_ok=True)

    well_name = tif_path.stem

    cfg = SamRunConfig(
        lower_area=lower_area,
        upper_area=upper_area,
        save_overlay=True,
        save_label_image=True,
        save_individual_masks=True,
    )

    print(f"🔬 Processing one well: {tif_path.name}")

    sam = load_sam(checkpoint, model_type=model_type, device=device)
    mask_gen = make_mask_generator(sam)

    rgb = read_morphology_tif_rgb_3072(tif_path)

    H, W = rgb.shape[:2]

    # ---------- detect/reuse arena circle (mirrors heartbeat reusable_cache) ----------
    reusable_well_dir = None
    if str(reusable_parent_dir).strip() != "":
        reusable_well_dir = Path(reusable_parent_dir) / output_root.name / well_name
        reusable_well_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] reusable artifact dir: {reusable_well_dir}")
    ring_params_path = (
        reusable_well_dir / "ring_params.json"
        if reusable_well_dir is not None else None
    )

    res = None
    reused_ring_params = False
    if ring_params_path is not None and ring_params_path.exists():
        try:
            with open(ring_params_path, "r") as _f:
                ring_cache = json.load(_f)
            cached_h = int(ring_cache.get("frame_h", H))
            cached_w = int(ring_cache.get("frame_w", W))
            cx_c = float(ring_cache["circle_cx"])
            cy_c = float(ring_cache["circle_cy"])
            r_c = float(ring_cache["circle_radius"])
            rmse_c = float(ring_cache.get("circle_rmse", np.nan))
            if (
                cached_h == int(H)
                and cached_w == int(W)
                and np.isfinite(cx_c)
                and np.isfinite(cy_c)
                and np.isfinite(r_c)
                and r_c >= min_ring_radius
            ):
                res = {"cx": cx_c, "cy": cy_c, "radius": r_c, "rmse": rmse_c}
                reused_ring_params = True
                print(f"[INFO] reused ring params from: {ring_params_path}")
            else:
                print(
                    f"[WARN] ignoring incompatible cached ring ({ring_params_path}): "
                    f"frame={cached_w}x{cached_h}, center=({cx_c}, {cy_c}), "
                    f"radius={r_c}; expected frame={W}x{H}, finite center, "
                    f"and radius >= {min_ring_radius:g} px. Running fresh detection."
                )
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
            min_radius=min_ring_radius,
        )

    x0, y0 = center_crop_origin_from_detected_center(
        cx=res["cx"],
        cy=res["cy"],
        frame_w=W,
        frame_h=H,
        crop_size=crop_size,
    )

    crop = rgb[y0:y0 + crop_size, x0:x0 + crop_size]

    print(
        f"arena center=({res['cx']:.1f}, {res['cy']:.1f}), "
        f"radius={res['radius']:.1f}, crop origin x0={x0}, y0={y0}"
    )

    RADIUS_PAD = 50
    # Persist the exact detected circle (global + crop-png coords) for reruns and analysis.
    if ring_params_path is not None and not reused_ring_params:
        try:
            with open(ring_params_path, "w") as _f:
                json.dump(
                    {
                        "well": well_name,
                        "plate": output_root.name,
                        "frame_h": int(H),
                        "frame_w": int(W),
                        "circle_cx": float(res["cx"]),
                        "circle_cy": float(res["cy"]),
                        "circle_radius": float(res["radius"]),
                        "circle_rmse": float(res["rmse"]) if res.get("rmse") is not None else None,
                        "x0": int(x0),
                        "y0": int(y0),
                        "crop_size": int(crop_size),
                        "cx_crop": float(res["cx"]) - x0,   # center in frame-png coords
                        "cy_crop": float(res["cy"]) - y0,
                        "radius_pad": RADIUS_PAD,
                        "min_ring_radius": float(min_ring_radius),
                    },
                    _f,
                    indent=2,
                )
            print(f"[INFO] saved ring params: {ring_params_path}")
        except Exception as e:
            print(f"[WARN] failed to save ring params ({ring_params_path}): {e}")

    crop_for_sam, arena_mask = apply_arena_only_to_crop(
        crop_rgb=crop,
        cx_global=res["cx"],
        cy_global=res["cy"],
        radius=res["radius"],
        x0=x0,
        y0=y0,
        radius_pad=RADIUS_PAD,   # tune this
        outside_value=0,         # black outside arena
    )
    #result = run_sam_on_rgb(mask_gen, crop, cfg,out_dir=output_root, base_name=well_name,)
    result = run_sam_on_rgb(
        mask_gen,
        crop_for_sam,
        cfg,
        out_dir=output_root,
        base_name=well_name,
        circle_mask=arena_mask,   # activate 30% arena-boundary rejection (rejects wall crescents)
        select_by="hier_darkeye", # dark-eye filter -> max SAM predicted_iou (replaces fishiness)
        raw_rgb=crop,             # dark5 read from the raw (non arena-blacked) crop
    )

    
   
    for m in result["masks"]:
        m["segmentation"] = m["segmentation"] & arena_mask
        m["area"] = int(np.count_nonzero(m["segmentation"]))
    
    
    
    masks = result.get("masks", [])

    severity_val = None
    do_skel = should_skeletonize_filtered(result)

    if do_skel and len(masks) > 0:
        # same logic as your original script: use first filtered mask
        m0 = masks[0]

        #sk = skeletonize_mask(m0["segmentation"])
        sk_out = skeletonize_mask(m0["segmentation"])
        save_skeleton_on_binary_figure(
        out_dir=output_root / "original",
        base_name=well_name,
        frame_idx=0,
        sk=sk_out,
    )

        sk_raw = sk_out.skeleton
        # sk_clean = keep_longest_skeleton_path(sk_raw)
        sk_long = keep_longest_skeleton_path(sk_raw)
        sk_clean = trim_skeleton_ends(sk_long, trim_frac=0.05)

        ys, xs = np.where(sk_clean > 0)
        midline_points = np.column_stack((xs, ys)).astype(np.int32)

        sk = SkeletonOutput(
            fish_binary=sk_out.fish_binary,
            skeleton=(sk_clean.astype(np.uint8) * 255),
            midline_points=midline_points,
        )

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
       

        save_skeleton_outputs(
            out_dir=output_root,
            base_name=well_name,
            frame_idx=0,
            sk=sk,
            overlay_bgr=crop_bgr,
            save_points_npy=True,
        )

        fit = fit_midline_and_severity(sk, degree=4, n_fit=100)
        severity_val = fit.severity
        print(f"🧠 {well_name} severity = {severity_val:.6f}")

        draw_and_save_poly_midline(
            out_dir=output_root,
            base_name=well_name,
            frame_idx=0,
            image_bgr=crop_bgr,
            fit_original=fit.fit_original,
            color=(255, 255, 255),
            thickness=6,
        )

        save_skeleton_on_binary_figure(
            out_dir=output_root,
            base_name=well_name,
            frame_idx=0,
            sk=sk,
            zoom=True,
            pad=25,
        )

        save_polyfit_debug_figure(
            out_dir=output_root,
            base_name=well_name,
            frame_idx=0,
            fit=fit,
            #ylim=(-400, 400),
        )

    else:
        print(f"⏭️ Skip skeleton/fit: {well_name} has {len(masks)} masks after filtering")

    rows = []

    for mask_id, m in enumerate(masks, start=1):
        area_px = int(m.get("area", np.count_nonzero(m["segmentation"])))
        mask_name = f"{well_name}_f{0:04d}_mask_{mask_id:03d}"

        rows.append({
            "tif_file": tif_path.name,
            "base_name": well_name,
            "frame_idx": 0,
            "mask_id": mask_id,
            "mask_name": mask_name,
            "area_px": area_px,
            "predicted_iou": m.get("predicted_iou"),
            "stability_score": m.get("stability_score"),
            "crop_box": m.get("bbox"),
            "point_coords": m.get("point_coords"),
            "severity": severity_val,
            "crop_x0": x0,
            "crop_y0": y0,
            "n_masks_after_filtering": len(masks),
            "skeletonized": bool(do_skel and len(masks) > 0),
        })

    save_results(
        out_dir=output_root,
        base_name=well_name,
        frame_idx=0,
        rgb=crop,
        result=result,
        cfg=cfg,
    )

    out_csv_dir = output_root / "csv"
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    csv_path = out_csv_dir / f"{well_name}_mask_areas.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Done: {well_name}")
    print(f"🧾 CSV saved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM + old skeleton pipeline for one well TIF."
    )

    parser.add_argument(
        "--tif_path",
        required=True,
        help="Path to one well TIF, e.g. /plate/well_A1.tif",
    )

    parser.add_argument(
        "--output_root",
        required=True,
        help="Output folder for this well or plate",
    )

    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--model_type", default=MODEL_TYPE_DEFAULT)
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    parser.add_argument("--crop_size", type=int, default=2048)
    parser.add_argument(
        "--min_ring_radius",
        type=float,
        default=800.0,
        help="Minimum fitted arena radius in pixels (inclusive; also checked for cached rings).",
    )
    parser.add_argument("--lower_area", type=int, default=40000)
    parser.add_argument("--upper_area", type=int, default=250000)
    parser.add_argument(
        "--reusable_parent_dir",
        type=str,
        default="",
        help=(
            "Optional parent folder for reusable artifacts (arena ring params) "
            "reused across reruns and by downstream analysis."
        ),
    )

    args = parser.parse_args()
    if not np.isfinite(args.min_ring_radius) or args.min_ring_radius <= 0:
        parser.error("--min_ring_radius must be positive and finite")

    process_one_well(
        tif_path=Path(args.tif_path),
        output_root=Path(args.output_root),
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        crop_size=args.crop_size,
        lower_area=args.lower_area,
        upper_area=args.upper_area,
        reusable_parent_dir=args.reusable_parent_dir,
        min_ring_radius=args.min_ring_radius,
    )


if __name__ == "__main__":
    main()