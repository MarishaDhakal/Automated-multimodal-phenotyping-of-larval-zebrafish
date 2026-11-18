# fluro_utils.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
from skimage.filters import gaussian
from skimage.morphology import disk, opening, closing, remove_small_objects
from skimage.measure import label, regionprops, find_contours, CircleModel
from skimage.util import img_as_float32
import cv2


@dataclass(frozen=True)
class CircleCropParams:
    """Crop window + detected inner circle parameters (all in original image coordinates)."""
    x0: int
    y0: int
    cx: float
    cy: float
    r: float
    crop_size: int
    bin_offset: float


# -----------------------------
# Binmask from green: baseline + offset
# -----------------------------
def make_green_binmask_fixed(
    frame_rgb: np.ndarray,
    *,
    bin_offset: float = 8.0,
    inset: int = 40,
    blur_sigma: float = 2.0,
    morph_radius: int = 6,
    min_obj_frac: float = 0.001,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Create a binary mask (uint8 0/255) highlighting bright green ring/glare.
    Threshold = median(center+corners) + bin_offset

    Returns:
      binmask_u8, info dict {baseline_255, thr_255}
    """
    H, W = frame_rgb.shape[:2]

    # green channel in 0..1
    G = img_as_float32(frame_rgb[..., 1])
    Gs = gaussian(G, sigma=blur_sigma, preserve_range=True)

    inset = int(inset)
    inset = max(1, min(inset, min(H, W) // 4))

    samples = np.array([
        Gs[H // 2, W // 2],
        Gs[inset, inset],
        Gs[inset, W - 1 - inset],
        Gs[H - 1 - inset, inset],
        Gs[H - 1 - inset, W - 1 - inset],
    ], dtype=np.float32)

    baseline = float(np.median(samples))              # 0..1
    thr = baseline + float(bin_offset) / 255.0        # 0..1

    bw = (Gs >= thr)

    se = disk(int(morph_radius))
    bw = opening(bw, se)
    bw = closing(bw, se)

    min_size = int(min_obj_frac * bw.size)
    bw = remove_small_objects(bw, min_size=max(10, min_size))

    binmask_u8 = (bw.astype(np.uint8) * 255)

    info = {
        "baseline_255": baseline * 255.0,
        "thr_255": thr * 255.0,
    }
    return binmask_u8, info


# -----------------------------
# Inner component from inverted binmask
# -----------------------------
def inner_component_from_binmask(
    binmask_u8: np.ndarray,
    *,
    morph_radius: int = 8,
    min_area_frac: float = 0.01,
) -> Optional[np.ndarray]:
    """
    Invert binmask (ring->white) so interior becomes foreground, then pick the component
    containing the image center (or largest if center is background).

    Returns:
      comp_bool (H,W) boolean mask, or None
    """
    H, W = binmask_u8.shape[:2]
    inv = (binmask_u8 == 0)

    se = disk(int(morph_radius))
    inv = opening(inv, se)
    inv = closing(inv, se)

    lab = label(inv)
    if lab.max() == 0:
        return None

    cy0, cx0 = H // 2, W // 2
    target = lab[cy0, cx0]

    props = regionprops(lab)
    if not props:
        return None

    if target == 0:
        p = max(props, key=lambda x: x.area)
    else:
        p = next((pp for pp in props if pp.label == target), None)
        if p is None:
            return None

    if p.area < int(min_area_frac * H * W):
        return None

    return (lab == p.label)

def save_ring_debug_image(
    frame_rgb: np.ndarray,
    circle_params,
    out_dir: Path,
    base_name: str,
):
    ring_dir = Path(out_dir) / "ring_debug"
    ring_dir.mkdir(parents=True, exist_ok=True)

    vis = frame_rgb.copy()

    cx = int(round(circle_params.cx))
    cy = int(round(circle_params.cy))
    r = int(round(circle_params.r))

    # draw detected ring/circle
    cv2.circle(vis, (cx, cy), r, (255, 0, 0), 4)

    # draw crop box
    x0 = int(circle_params.x0)
    y0 = int(circle_params.y0)
    s = int(circle_params.crop_size)

    cv2.rectangle(
        vis,
        (x0, y0),
        (x0 + s, y0 + s),
        (0, 255, 0),
        3,
    )

    # center point
    cv2.circle(vis, (cx, cy), 8, (255, 255, 0), -1)

    cv2.imwrite(
        str(ring_dir / f"{base_name}_ring_detection.png"),
        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
    )

# -----------------------------
# Circle fit from component boundary
# -----------------------------
def fit_circle_from_component(
    comp_bool: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """
    Fit a circle to a boolean component boundary using skimage CircleModel.

    Returns:
      (cx, cy, r) in pixel coords, or None
    """
    contours = find_contours(comp_bool.astype(np.float32), level=0.5)
    if not contours:
        return None

    cnt = max(contours, key=lambda c: c.shape[0])  # (row, col)
    xs = cnt[:, 1]
    ys = cnt[:, 0]
    pts = np.column_stack([xs, ys])

    model = CircleModel()
    if not model.estimate(pts):
        return None

    cx, cy, r = model.params
    return float(cx), float(cy), float(r)


# -----------------------------
# Crop helpers
# -----------------------------
def crop_square_from_center(
    rgb: np.ndarray,
    cx: float,
    cy: float,
    *,
    size: int = 1048,
) -> Tuple[np.ndarray, int, int]:
    """
    Crop fixed square size around (cx,cy).
    Returns: (crop_rgb, x0, y0) where (x0,y0) is top-left in original image.
    """
    H, W = rgb.shape[:2]
    half = size // 2

    x1 = int(round(cx - half)); x2 = int(round(cx + half))
    y1 = int(round(cy - half)); y2 = int(round(cy + half))

    # shift to keep in bounds (preserve size)
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > W:
        s = x2 - W
        x1 -= s; x2 = W
    if y2 > H:
        s = y2 - H
        y1 -= s; y2 = H

    x1 = max(0, x1); y1 = max(0, y1)
    crop = rgb[y1:y2, x1:x2].copy()
    return crop, x1, y1


# -----------------------------
# One-shot: from rgb -> CircleCropParams
# -----------------------------
def estimate_circle_crop_params(
    frame_rgb: np.ndarray,
    *,
    crop_size: int = 1048,
    bin_offset: float = 8.0,
    bin_inset: int = 40,
    bin_blur_sigma: float = 2.0,
    bin_morph_radius: int = 6,
    inner_morph_radius: int = 8,
) -> Tuple[CircleCropParams, np.ndarray, Dict[str, float]]:
    """
    Full pipeline:
      rgb -> binmask -> invert -> inner comp -> circle fit -> crop params

    Returns:
      (CircleCropParams, binmask_u8, info)
    Raises:
      RuntimeError with a short reason if it fails
    """
    binmask, info = make_green_binmask_fixed(
        frame_rgb,
        bin_offset=bin_offset,
        inset=bin_inset,
        blur_sigma=bin_blur_sigma,
        morph_radius=bin_morph_radius,
    )

    comp = inner_component_from_binmask(
        binmask,
        morph_radius=inner_morph_radius,
    )
    if comp is None:
        raise RuntimeError("inner_component_not_found")

    circle = fit_circle_from_component(comp)
    if circle is None:
        raise RuntimeError("circle_fit_failed")

    cx, cy, r = circle
    print(f"[DEBUG] raw circle: cx={cx:.1f}, cy={cy:.1f}, r={r:.1f}")
    MIN_RADIUS = 100

    if r < MIN_RADIUS:
        H, W = frame_rgb.shape[:2]

        # fallback to image center
        cx = W / 2.0
        cy = H / 2.0
        r = float(MIN_RADIUS)

    # compute crop top-left (x0,y0) by cropping once
    _, x0, y0 = crop_square_from_center(frame_rgb, cx, cy, size=crop_size)

    params = CircleCropParams(
        x0=int(x0),
        y0=int(y0),
        cx=float(cx),
        cy=float(cy),
        r=float(r),
        crop_size=int(crop_size),
        bin_offset=float(bin_offset),
    )
    return params, binmask, info


def apply_circle_crop(
    frame_rgb: np.ndarray,
    params: CircleCropParams,
) -> Tuple[np.ndarray, int, int]:
    """
    Fast crop using stored params (no re-detection).
    Returns: (crop_rgb, x0, y0)
    """
    x0, y0, s = params.x0, params.y0, params.crop_size
    crop = frame_rgb[y0:y0 + s, x0:x0 + s].copy()
    return crop, int(x0), int(y0)


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

def remove_fish_touching_ring_from_band(ring, expected_center, tol=35):
    ys, xs = np.nonzero(ring)
    if len(xs) < 50:
        return ring

    cx0, cy0 = expected_center
    rr = np.sqrt((xs - cx0)**2 + (ys - cy0)**2)

    # dominant radius = where most boundary pixels are
    hist, edges = np.histogram(rr, bins=80)
    k = np.argmax(hist)
    r_mode = 0.5 * (edges[k] + edges[k + 1])

    keep = np.abs(rr - r_mode) <= tol

    clean = np.zeros_like(ring, dtype=bool)
    clean[ys[keep], xs[keep]] = True
    return clean
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
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {rgb.shape}")

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    #gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = rgb[..., 1].copy()
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
        binary = (gray >= thr).astype(np.uint8) * 255

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

            # ring = make_thin_ring_from_component(
            #     comp,
            #     inner_band_low=inner_band_low,
            #     inner_band_high=inner_band_high,
            # )
            ring = inner_boundary_from_annulus_component(
                comp_bool=(comp > 0),
                expected_center=expected_center,
                band_px=6,
            )
            ring_orig = ring.copy()

            ring_clean = remove_fish_touching_ring_from_band(
                ring,
                expected_center=expected_center,
                tol=100,
            )

            # If cleanup is too destructive → revert
            if np.count_nonzero(ring_clean) < 0.6 * np.count_nonzero(ring_orig):
                ring = ring_orig
            else:
                ring = ring_clean
            

            ring_pixels = int(ring.sum())
            if ring_pixels < min_ring_pixels:
                continue

            try:
                cx, cy, radius = fit_circle_from_mask(ring)
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

def inner_boundary_from_annulus_component(comp_bool, expected_center, band_px=6):
    comp_u8 = comp_bool.astype(np.uint8)

    # distance from background into white annulus
    dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)

    # thin boundary band of component
    boundary = (dist > 0) & (dist <= band_px)

    ys, xs = np.nonzero(boundary)
    if len(xs) < 50:
        return boundary

    cx0, cy0 = expected_center
    rr = np.sqrt((xs - cx0)**2 + (ys - cy0)**2)

    # keep inner half of boundary points
    cutoff = np.percentile(rr, 50)
    inner_boundary = np.zeros_like(boundary, dtype=bool)
    inner_boundary[ys[rr <= cutoff], xs[rr <= cutoff]] = True

    return inner_boundary

def estimate_circle_crop_params_annulus(
    frame_rgb: np.ndarray,
    *,
    crop_size: int = 1048,
    radius_override: float | None = None,
    annulus_min_area: int | None = None,
) -> tuple[CircleCropParams, np.ndarray, dict]:
    """
    HB-style annulus detector adapted for fluorescence pipeline.
    Returns:
        CircleCropParams, ring_mask, info
    """

    # res = detect_ring_auto_prefer_annulus(
    #     frame_rgb,
    #     p_range=range(50, 60),
    #     morph_ksize=1,
    #     min_area=100000,
    #     max_fill_ratio=0.75,
    #     min_ring_pixels=200,
    #     min_coverage_rad=1.5 * np.pi,
    #     inner_band_low=2.0,
    #     inner_band_high=6.0,
    #     expected_center=(frame_rgb.shape[1] / 2.0, frame_rgb.shape[0] / 2.0),
    #     center_weight=0.002,
    # )
    best_res = None
    if annulus_min_area is None:
        if crop_size <= 1048:
            annulus_min_area = 100000
        elif crop_size >= 2048:
            annulus_min_area = 1000000
        else:
            annulus_min_area = 100000

    for k in [3, 5, 7, 9, 11, 1, 13]:
        try:
            res = detect_ring_auto_prefer_annulus(
                frame_rgb,
                p_range=range(50, 60),
                morph_ksize=k,
                min_area=int(annulus_min_area),
                max_fill_ratio=0.75,
                min_ring_pixels=200,
                min_coverage_rad=1.5 * np.pi,
                inner_band_low=2.0,
                inner_band_high=6.0,
                center_weight=0.002,
            )

            res["morph_ksize"] = k

            if best_res is None or res["score"] > best_res["score"]:
                best_res = res

        except RuntimeError:
            continue

    if best_res is None:
        raise RuntimeError("Could not detect ring for any morph_ksize")

    res = best_res

    H, W = frame_rgb.shape[:2]

    cx = float(res["cx"])
    cy = float(res["cy"])
    r_detected  = float(res["radius"])
    if radius_override is not None:
        print(f"[INFO] annulus radius override: detected {r_detected:.1f} -> {radius_override:.1f}")
        r = float(radius_override)
    else:
        r = float(r_detected)
    info = dict(res)    
    info["radius_detected"] = float(r_detected)
    info["radius_used"] = float(r)
    info["radius_override"] = np.nan if radius_override is None else float(radius_override)



    _, x0, y0 = crop_square_from_center(
        frame_rgb,
        cx,
        cy,
        size=crop_size,
    )

    params = CircleCropParams(
        x0=int(x0),
        y0=int(y0),
        cx=float(cx),
        cy=float(cy),
        r=float(r),
        crop_size=int(crop_size),
        bin_offset=np.nan,
    )

    #return params, res["ring_mask"].astype(np.uint8) * 255, res
    return params, res["ring_mask"].astype(np.uint8) * 255, info