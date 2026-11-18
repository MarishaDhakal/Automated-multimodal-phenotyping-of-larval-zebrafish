# sam_pipeline/skeletonize_utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import cv2
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt



# Data containers

@dataclass
class SkeletonOutput:
    fish_binary: np.ndarray        # uint8 0/255 (H,W)
    skeleton: np.ndarray           # uint8 0/255 (H,W)
    midline_points: np.ndarray     # (N,2) int (x,y)


@dataclass
class FitOutput:
    rotated_points: np.ndarray      # (N,2) rotated skeleton points (x',y')
    center: np.ndarray              # (2,) center (x,y)
    cos: float
    sin: float

    x_sorted: np.ndarray            # (N,)
    y_sorted: np.ndarray            # (N,)
    lin_connector: np.ndarray       # (M,2) line between endpoints in rotated space

    fit_rotated: np.ndarray         # (M,2) poly fit curve in rotated space
    fit_original: np.ndarray        # (M,2) poly fit curve in original coords

    severity: float                 # curvature severity metric



# Skeletonize 

def skeletonize_mask(mask_segmentation: np.ndarray) -> SkeletonOutput:
    """
    Notebook-equivalent skeletonization:

      fish_binary      = mask.astype(uint8) * 255
      fish_binary_bool = fish_binary // 255
      skeleton         = skeletonize(fish_binary_bool).astype(uint8) * 255
      midline_points   = np.where(skeleton>0) as (x,y)

    mask_segmentation: bool or 0/1 array (H,W)
    """
    fish_binary = mask_segmentation.astype(np.uint8) * 255
    fish_binary_bool = fish_binary // 255

    skel = skeletonize(fish_binary_bool).astype(np.uint8) * 255

    ys, xs = np.where(skel > 0)
    midline_points = np.column_stack((xs, ys)).astype(np.int32)

    return SkeletonOutput(
        fish_binary=fish_binary,
        skeleton=skel,
        midline_points=midline_points,
    )


def should_skeletonize_filtered(result: Dict[str, Any]) -> bool:
    """
    Only skeletonize if the *filtered* mask list (result['masks']) has exactly 1 mask.
    """
    masks_f = result.get("masks", [])
    return len(masks_f) == 1


# Fit midline + severity (PCA rotate + polyfit)
def rotate_major_axis(midline_points: np.ndarray):
    """
    PCA/SVD to align points along major axis (same as your notebook).
    Returns rotated_points, center, cos, sin (for undo rotation).
    """
    if midline_points.shape[0] < 5:
        raise ValueError("Not enough skeleton points for PCA/fit.")

    center = midline_points.mean(axis=0)
    centered = midline_points - center

    # SVD on centered points
    _, _, Vt = np.linalg.svd(centered, full_matrices=True)
    major_axis = Vt[0]

    angle = np.arctan2(major_axis[1], abs(major_axis[0]))

    # Rotate to align major axis with x-axis
    cos, sin = np.cos(-angle), np.sin(-angle)
    rot_matrix = np.array([[cos, sin], [-sin, cos]], dtype=np.float64)

    rotated = centered @ rot_matrix.T
    return rotated, center, float(cos), float(sin)


def sort_skeleton(rotated: np.ndarray, n_line: int = 100):
    """
    Sort by x-coordinate for polynomial fitting and create linear connector.
    """
    sort_idx = np.argsort(rotated[:, 0])
    x_sorted = rotated[sort_idx, 0]
    y_sorted = rotated[sort_idx, 1]

    # Connect endpoints linearly
    x0, y0 = x_sorted[0], y_sorted[0]
    xn, yn = x_sorted[-1], y_sorted[-1]
    lin_connector = np.linspace([x0, y0], [xn, yn], num=n_line)

    return x_sorted, y_sorted, lin_connector


def fit_poly(x_sorted: np.ndarray, y_sorted: np.ndarray, degree: int = 4, n_fit: int = 100):
    """
    Fit polynomial y(x) in rotated space.
    """
    if x_sorted.size < degree + 2:
        raise ValueError(f"Not enough points for degree={degree} polynomial fit.")

    coeffs = np.polyfit(x_sorted, y_sorted, degree)
    p = np.poly1d(coeffs)

    x_fit = np.linspace(x_sorted.min(), x_sorted.max(), n_fit)
    y_fit = p(x_fit)

    fit_rotated = np.column_stack((x_fit, y_fit))
    return fit_rotated


def undo_rotation(fit_rotated: np.ndarray, center: np.ndarray, cos: float, sin: float):
    """
    Undo PCA rotation to original coordinates.
    """
    rot_back = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    fit_original = (fit_rotated @ rot_back.T) + center
    return fit_original


def curvature_severity(fit_rotated: np.ndarray, lin_connector: np.ndarray) -> float:
    """
    Notebook-equivalent severity:
      integral(|y_fit - y_line| dx) / distance(endpoints)
    """
    x_fit = fit_rotated[:, 0]
    y_fit = fit_rotated[:, 1]

    # Ensure connector length matches
    if lin_connector.shape[0] != fit_rotated.shape[0]:
        lin_connector = np.linspace(lin_connector[0], lin_connector[-1], num=fit_rotated.shape[0])

    y_line = lin_connector[:, 1]

    integral = np.trapz(np.abs(y_fit - y_line), x_fit)
    normalizer = np.hypot(x_fit[-1] - x_fit[0], y_fit[-1] - y_fit[0])

    if normalizer == 0:
        return 0.0

    return float(integral / normalizer)


def fit_midline_and_severity(
    sk: SkeletonOutput,
    degree: int = 4,
    n_fit: int = 100,
) -> FitOutput:
    """
    Full "fit fish" step:
    - PCA rotate skeleton points
    - sort points
    - polyfit (degree 4)
    - undo rotation
    - compute severity
    """
    midline_points = sk.midline_points.astype(np.float64)

    rotated, center, cos, sin = rotate_major_axis(midline_points)
    x_sorted, y_sorted, lin_connector = sort_skeleton(rotated, n_line=n_fit)
    fit_rotated = fit_poly(x_sorted, y_sorted, degree=degree, n_fit=n_fit)
    fit_original = undo_rotation(fit_rotated, center, cos, sin)
    severity = curvature_severity(fit_rotated, lin_connector)

    return FitOutput(
        rotated_points=rotated,
        center=center,
        cos=cos,
        sin=sin,
        x_sorted=x_sorted,
        y_sorted=y_sorted,
        lin_connector=lin_connector,
        fit_rotated=fit_rotated,
        fit_original=fit_original,
        severity=severity,
    )
def center_crop_origin_from_detected_center(cx, cy, frame_w, frame_h, crop_size=1048):
    cx_i = int(round(cx))
    cy_i = int(round(cy))

    xoff = cx_i - crop_size // 2
    yoff = cy_i - crop_size // 2

    xoff = max(0, min(xoff, frame_w - crop_size))
    yoff = max(0, min(yoff, frame_h - crop_size))
    return int(xoff), int(yoff)


def make_thin_ring_from_component(comp_u8, inner_band_low=2.0, inner_band_high=6.0):
    dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
    ring = (dist >= inner_band_low) & (dist <= inner_band_high)
    return ring

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
    min_radius=800.0,
):
    if not np.isfinite(min_radius) or min_radius <= 0:
        raise ValueError("min_radius must be positive and finite")
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
                rmse = circle_fit_rmse(ring, cx, cy, radius)
            except Exception:
                continue

            if not np.isfinite(radius) or radius < min_radius:
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
        raise RuntimeError(
            f"Could not detect a good annular ring candidate with radius >= {min_radius:g} px"
        )

    return best



# Saving utilities

def save_skeleton_outputs(
    out_dir: Path,
    base_name: str,
    frame_idx: int,
    sk: SkeletonOutput,
    overlay_bgr: Optional[np.ndarray] = None,
    save_points_npy: bool = True,
) -> Dict[str, Path]:
    """
    Saves into:
      out_dir/skeletons/
        <base>_f0000_fish_binary.png
        <base>_f0000_skeleton.png
        <base>_f0000_skeleton_overlay.png   (if overlay_bgr provided)
        <base>_f0000_midline_points.npy     (optional)
    """
    out_dir = Path(out_dir)
    skel_dir = out_dir / "skeletons"
    skel_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    p_bin = skel_dir / f"{base_name}_f{frame_idx:04d}_fish_binary.png"
    cv2.imwrite(str(p_bin), sk.fish_binary)
    paths["fish_binary_png"] = p_bin

    p_skel = skel_dir / f"{base_name}_f{frame_idx:04d}_skeleton.png"
    cv2.imwrite(str(p_skel), sk.skeleton)
    paths["skeleton_png"] = p_skel

    if overlay_bgr is not None:
        ov = overlay_bgr.copy()
        ov[sk.skeleton > 0] = (255, 255, 255)
        p_ov = skel_dir / f"{base_name}_f{frame_idx:04d}_skeleton_overlay.png"
        cv2.imwrite(str(p_ov), ov)
        paths["skeleton_overlay_png"] = p_ov

    if save_points_npy:
        p_pts = skel_dir / f"{base_name}_f{frame_idx:04d}_midline_points.npy"
        np.save(p_pts, sk.midline_points)
        paths["midline_points_npy"] = p_pts

    return paths

def center_crop_origin_from_detected_center(cx, cy, frame_w, frame_h, crop_size=1048):
    cx_i = int(round(cx))
    cy_i = int(round(cy))

    xoff = cx_i - crop_size // 2
    yoff = cy_i - crop_size // 2

    xoff = max(0, min(xoff, frame_w - crop_size))
    yoff = max(0, min(yoff, frame_h - crop_size))
    return int(xoff), int(yoff)

def draw_and_save_poly_midline(
    out_dir: Path,
    base_name: str,
    frame_idx: int,
    image_bgr: np.ndarray,
    fit_original: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 6,
) -> Path:
    """
    Draw polynomial midline onto an image and save into:
      out_dir/fits/<base>_f0000_poly_midline.png
    """
    out_dir = Path(out_dir)
    fits_dir = out_dir / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)

    img = image_bgr.copy()
    pts = np.round(fit_original).astype(np.int32).reshape((-1, 1, 2))

    h, w = img.shape[:2]
    pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, w - 1)
    pts[:, 0, 1] = np.clip(pts[:, 0, 1], 0, h - 1)

    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

    out_path = fits_dir / f"{base_name}_f{frame_idx:04d}_poly_midline.png"
    cv2.imwrite(str(out_path), img)
    return out_path

def _zoom_to_binary(ax, binary_255: np.ndarray, pad: int = 25) -> None:
    ys, xs = np.where(binary_255 > 0)
    if xs.size == 0:
        return

    h, w = binary_255.shape[:2]
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, w - 1)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, h - 1)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # invert y for image coordinates



# def save_skeleton_on_binary_figure(
#     out_dir: Path,
#     base_name: str,
#     frame_idx: int,
#     sk: SkeletonOutput,
#     zoom: bool = True,
#     pad: int = 25,
# ) -> Path:
#     out_dir = Path(out_dir)
#     fig_dir = out_dir / "figures_skeleton"
#     fig_dir.mkdir(parents=True, exist_ok=True)

#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))

#     # notebook-style overlay order
#     ax.imshow(sk.fish_binary, cmap="gray", interpolation="nearest")
#     ax.imshow(sk.skeleton, cmap="gray", interpolation="nearest", alpha=0.9)

#     if sk.midline_points.size > 0:
#         ax.scatter(
#             sk.midline_points[:, 0],
#             sk.midline_points[:, 1],
#             c="orange",
#             s=5,
#             alpha=0.5,
#         )

#     ax.set_title("Skeleton Overlayed on Fish Binary")
#     ax.axis("off")

#     if zoom:
#         _zoom_to_binary(ax, sk.fish_binary, pad=pad)

#     out_path = fig_dir / f"{base_name}_f{frame_idx:04d}_skeleton_on_binary.png"
#     fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
#     plt.close(fig)
#     return out_path

def save_skeleton_on_binary_figure(
    out_dir: Path,
    base_name: str,
    frame_idx: int,
    sk: SkeletonOutput,
    zoom: bool = True,
    pad: int = 25,
) -> Path:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures_skeleton"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # --- fish binary as true RGB: black bg + white fish ---
    h, w = sk.fish_binary.shape
    binary_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    binary_rgb[sk.fish_binary > 0] = (255, 255, 255)
    ax.imshow(binary_rgb, interpolation="nearest")

    # --- skeleton drawn as points (does NOT gray out the background) ---
    ys, xs = np.where(sk.skeleton > 0)
    if xs.size > 0:
        ax.scatter(xs, ys, c="lightgray", s=1, alpha=0.6)

    # --- midline in orange (points + connecting line) ---
    if sk.midline_points.size > 0:
        pts = sk.midline_points
        #ax.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=2)
        ax.scatter(pts[:, 0], pts[:, 1], c="orange", s=5, alpha=0.5)

    ax.set_title("Skeleton + Midline on Fish Binary")
    ax.axis("off")

    if zoom:
        _zoom_to_binary(ax, sk.fish_binary, pad=pad)

    out_path = fig_dir / f"{base_name}_f{frame_idx:04d}_skeleton_on_binary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path



# def save_polyfit_debug_figure(
#     out_dir: Path,
#     base_name: str,
#     frame_idx: int,
#     fit: FitOutput,
#     ylim: tuple[float, float] = (-600, 600),
# ) -> Path:
#     out_dir = Path(out_dir)
#     fig_dir = out_dir / "figures_fit"
#     fig_dir.mkdir(parents=True, exist_ok=True)

#     fig, ax = plt.subplots(1, 1, figsize=(7, 5))

#     # notebook-style colors
#     ax.scatter(fit.x_sorted, fit.y_sorted, c="red", s=5, alpha=0.5, label="Skeleton")
#     ax.scatter(
#         fit.lin_connector[:, 0],
#         fit.lin_connector[:, 1],
#         c="green",
#         s=5,
#         alpha=0.5,
#         label="Linear Fit",
#     )

#     x_fit = fit.fit_rotated[:, 0]
#     y_fit = fit.fit_rotated[:, 1]
#     ax.plot(x_fit, y_fit, "b-", linewidth=2, label=f"Severity: {fit.severity:.3g}")

#     ax.set_title("Poly Fit")
#     ax.set_ylim(*ylim)
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     out_path = fig_dir / f"{base_name}_f{frame_idx:04d}_polyfit_debug.png"
#     #fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)
#     return out_path
def save_polyfit_debug_figure(
    out_dir: Path,
    base_name: str,
    frame_idx: int,
    fit: FitOutput,
    pad_frac: float = 0.10,
) -> Path:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures_fit"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # --- data ---
    x = fit.x_sorted
    y = fit.y_sorted

    x_fit = fit.fit_rotated[:, 0]
    y_fit = fit.fit_rotated[:, 1]

    # --- plots ---
    ax.scatter(x, y, c="red", s=5, alpha=0.5, label="Skeleton")
    ax.scatter(
        fit.lin_connector[:, 0],
        fit.lin_connector[:, 1],
        c="green",
        s=5,
        alpha=0.5,
        label="Linear Fit",
    )
    ax.plot(x_fit, y_fit, "b-", linewidth=2, label=f"Severity: {fit.severity:.3g}")

    # --- flexible limits with padding ---
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    dx = max(x_max - x_min, 1.0)
    dy = max(y_max - y_min, 1.0)

    pad_x = pad_frac * dx
    pad_y = pad_frac * dy

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # --- aesthetics ---
    ax.set_title("Poly Fit")
    ax.set_aspect("equal")   # preserves curvature shape
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- save ---
    out_path = fig_dir / f"{base_name}_f{frame_idx:04d}_polyfit_debug.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path

def save_segmented_fish_with_poly_midline(
    out_dir: Path,
    base_name: str,
    frame_idx: int,
    crop_rgb: np.ndarray,
    mask_segmentation: np.ndarray,
    fit_original: np.ndarray,
    zoom: bool = True,
    pad: int = 25,
    thickness: int = 6,
) -> Path:
    out_dir = Path(out_dir)
    fits_dir = out_dir / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)

    seg_rgb = crop_rgb.copy()
    seg_rgb[~mask_segmentation] = 0
    seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

    pts = np.round(fit_original).astype(np.int32).reshape((-1, 1, 2))
    h, w = seg_bgr.shape[:2]
    pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, w - 1)
    pts[:, 0, 1] = np.clip(pts[:, 0, 1], 0, h - 1)
    cv2.polylines(seg_bgr, [pts], isClosed=False, color=(255, 255, 255), thickness=thickness)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Segmented Fish + Polynomial Midline")
    ax.axis("off")

    if zoom:
        fish_binary = (mask_segmentation.astype(np.uint8) * 255)
        _zoom_to_binary(ax, fish_binary, pad=pad)

    out_path = fits_dir / f"{base_name}_f{frame_idx:04d}_segmented_poly_midline.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path
def make_plate_montage_with_blanks(
    img_dir: Path,
    filename_fn,
    out_name: str,
    rows: str = "ABCDEFGH",
    cols: list[int] | None = None,
    blank_value: int = 0,
    strict: bool = False,
) -> Path:
    """
    Create an 8x12 montage from per-well images in img_dir.
    Missing wells are filled with a blank tile to preserve plate structure.

    Parameters
    ----------
    img_dir : Path
        Folder containing images (may be incomplete).
    filename_fn : callable(row_char, col_int) -> str
        Function returning expected filename for each well.
    out_name : str
        Output montage filename (saved inside img_dir).
    blank_value : int
        Pixel value for blank tiles (0 => black).
    strict : bool
        If True, raise on any missing tile. If False, fill blanks.

    Returns
    -------
    Path to saved montage image.
    """
    img_dir = Path(img_dir)
    if cols is None:
        cols = list(range(1, 13))

    # Find the first existing image to set tile size
    first = None
    for r in rows:
        for c in cols:
            p = img_dir / filename_fn(r, c)
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    first = img
                    break
        if first is not None:
            break

    if first is None:
        raise RuntimeError(f"No images found in {img_dir} to build montage: {out_name}")

    # Normalize to 3-channel BGR
    if first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    elif first.shape[2] == 4:
        first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)

    tile_h, tile_w = first.shape[:2]
    blank = np.full((tile_h, tile_w, 3), blank_value, dtype=np.uint8)

    missing = []
    row_tiles_all = []

    for r in rows:
        row_tiles = []
        for c in cols:
            fname = filename_fn(r, c)
            p = img_dir / fname

            if not p.exists():
                if strict:
                    raise FileNotFoundError(f"Missing image: {p}")
                missing.append(fname)
                row_tiles.append(blank.copy())
                continue

            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                if strict:
                    raise RuntimeError(f"Failed to read image: {p}")
                missing.append(fname)
                row_tiles.append(blank.copy())
                continue

            # Normalize to 3-channel BGR
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Resize if mismatch
            if img.shape[0] != tile_h or img.shape[1] != tile_w:
                img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

            row_tiles.append(img)

        row_tiles_all.append(np.hstack(row_tiles))

    plate = np.vstack(row_tiles_all)

    out_path = img_dir / out_name
    cv2.imwrite(str(out_path), plate)

    if missing:
        print(f"🟨 Montage '{out_name}': filled {len(missing)} missing wells with blanks.")

    return out_path
def make_all_skeleton_related_montages(
    output_root: Path,
    frame_idx: int = 0,
) -> dict[str, Path]:
    """
    Build montages (with blanks) for:
      - fits: poly midline images
      - figures_skeleton: skeleton on binary
      - figures_fit: polyfit debug plot

    Returns dict of montage paths created.
    """
    output_root = Path(output_root)
    out: dict[str, Path] = {}

    # fits (poly midline)
    fits_dir = output_root / "fits"
    if fits_dir.exists():
        out["poly_midline"] = make_plate_montage_with_blanks(
            img_dir=fits_dir,
            filename_fn=lambda r, c: f"well_{r}{c}_f{frame_idx:04d}_poly_midline.png",
            out_name="plate_poly_midline_montage.png",
        )

    # figures_skeleton
    sk_dir = output_root / "figures_skeleton"
    if sk_dir.exists():
        out["skeleton_on_binary"] = make_plate_montage_with_blanks(
            img_dir=sk_dir,
            filename_fn=lambda r, c: f"well_{r}{c}_f{frame_idx:04d}_skeleton_on_binary.png",
            out_name="plate_skeleton_on_binary_montage.png",
        )

    # figures_fit
    fit_dir = output_root / "figures_fit"
    if fit_dir.exists():
        out["polyfit_debug"] = make_plate_montage_with_blanks(
            img_dir=fit_dir,
            filename_fn=lambda r, c: f"well_{r}{c}_f{frame_idx:04d}_polyfit_debug.png",
            out_name="plate_polyfit_debug_montage.png",
        )

    return out
