from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union, Dict, Any, List
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

import csv

import numpy as np
import cv2

try:
    import tifffile
except ImportError:
    tifffile = None

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

DEFAULT_SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT") or str(
    Path(__file__).resolve().parents[1] / "checkpoints" / "sam_vit_h_4b8939.pth"
)


def load_sam(
    checkpoint_path: str = DEFAULT_SAM_CHECKPOINT,
    model_type: str = "vit_h",
    device: str = "cuda",
) -> Any:
    checkpoint_path = str(checkpoint_path)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    return sam

def reject_edge_or_ring_mask(
    seg: np.ndarray,
    circle_mask: np.ndarray | None = None,
    band_px: int = 35,
    max_boundary_frac: float = 0.20,
) -> bool:
    seg = seg.astype(bool)

    if seg.sum() == 0:
        return True

    # square crop border
    # if touches_border(seg):
    #     return True
    if touches_corner(seg):
        return True
    

    # circular arena boundary
    if circle_mask is not None:
        circle_mask = circle_mask.astype(bool)

        dist_inside = cv2.distanceTransform(
            circle_mask.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        boundary_band = circle_mask & (dist_inside <= band_px)
        frac_boundary = (seg & boundary_band).sum() / seg.sum()

        if frac_boundary > max_boundary_frac:
            return True

    return False
def make_mask_generator(
    sam: Any,
    points_per_side=16,            # points^2 total masks (can increase with higher comp. power)
    pred_iou_thresh=0.8,          # Lower to include fish masks with moderate confidence
    stability_score_thresh=0.8,   # Lower to handle reflections or fuzzy edges
    min_mask_region_area=40000,    # Filter out small noise (e.g., reflections)
    crop_n_layers=0,              
    crop_nms_thresh=0.4,          # Low probability of overlapping masks
    points_per_batch=5           # Adjust for memory constraints if needed
    
) -> SamAutomaticMaskGenerator:
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        points_per_batch=points_per_batch,
    )

# def is_video(path: Union[str, Path]) -> bool:
#     return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


def is_tif(path: Union[str, Path]) -> bool:
    return str(path).lower().endswith((".tif", ".tiff"))

# ============================================================
# Input contracts / validation
# ============================================================
def assert_uint8_rgb(img: np.ndarray, name: str = "image") -> None:
    """
    Enforce strict contract for SAM input:
    - dtype uint8
    - 3 channels RGB (H,W,3)
    - values in [0,255]
    """
    if img.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8, got {img.dtype}")

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H,W,3), got {img.shape}")

    # With uint8 this is always 0..255, but keep as a sanity check
    mn, mx = int(img.min()), int(img.max())
    if mn < 0 or mx > 255:
        raise ValueError(f"{name} values must be in [0,255], got min={mn}, max={mx}")


def assert_fixed_size(img: np.ndarray, h: int, w: int, name: str = "image") -> None:
    if img.shape[0] != h or img.shape[1] != w:
        raise ValueError(f"{name} must be {h}x{w}, got {img.shape[:2]}")


# ============================================================
# Media loading (tif OR mp4)
# ============================================================
def read_tif_as_rgb_uint8(path: Union[str, Path]) -> np.ndarray:
    """
    Read a tif and return RGB uint8.
    NOTE: This function does NOT normalize or convert dtype.
          It will raise if tif is not already uint8.
    """
    if tifffile is None:
        raise ImportError("tifffile not installed. Run: pip install tifffile")

    img = tifffile.imread(str(path))

    # Enforce uint8 early (since we removed normalization)
    if img.dtype != np.uint8:
        raise TypeError(f"TIF must be uint8. Got {img.dtype} for {path}")

    # Convert grayscale -> RGB
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # Handle channels-first (C,H,W) for C in (1,3)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    # Ensure (H,W,3)
    if img.ndim != 3:
        raise ValueError(f"Unsupported TIF shape: {img.shape} for {path}")

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] >= 3:
        img = img[..., :3]
    else:
        raise ValueError(f"TIF channel dimension unexpected: {img.shape} for {path}")

    assert_uint8_rgb(img, name=f"tif_rgb({Path(path).name})")
    return img


def read_morphology_tif_rgb_3072(path: Union[str, Path]) -> np.ndarray:
    """
    Special loader for your morphology tif files (fixed 3072x3072).
    """
    rgb = read_tif_as_rgb_uint8(path)
    assert_fixed_size(rgb, 3072, 3072, name=f"morphology_tif({Path(path).name})")
    return rgb




def load_media_rgb(
    path: Union[str, Path],
    frame_idx: Optional[int] = None,
    resize_scale: Optional[float] = None,
    tif_fixed_3072: bool = False,
) -> List[Tuple[int, np.ndarray]]:
    """
    Convenience loader:
    - For TIF: returns [(0, image_rgb)]
      - If tif_fixed_3072=True, enforces 3072x3072.
    """
    path = Path(path)

    if is_tif(path):
        rgb = read_morphology_tif_rgb_3072(path) if tif_fixed_3072 else read_tif_as_rgb_uint8(path)
        return [(0, rgb)]


    raise ValueError(f"Unsupported file type: {path}")

# --- cropping image
def center_crop_rgb(img: np.ndarray, crop_size: int = 2048) -> tuple[np.ndarray, int, int]:
    """
    Center-crop an RGB image to (crop_size x crop_size).
    Returns: (crop, x0, y0) where (x0,y0) is the top-left corner in original image coords.
    """
    h, w, c = img.shape
    if c != 3:
        raise ValueError(f"Expected RGB (H,W,3). Got {img.shape}")

    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} bigger than image size {(h,w)}")

    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2

    crop = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
    return crop, x0, y0
import numpy as np
import cv2

def auto_p_dark_minimal(
    gray_u8: np.ndarray,
    p_min: int = 6,
    p_max: int = 14,
    ridge_min_points: int = 120,
    close_ksize: int = 13,
    open_ksize: int = 9,
    ridge_dilate_ksize: int = 9,
    ridge_dt_percentile: float = 70,
) -> int:
    """
    Minimal auto p_dark:
    Pick the *smallest* percentile p in [p_min..p_max] that yields enough ridge points.
    This tends to avoid including extra dark junk (shadows/fish) that higher p can add.
    """
    def odd(k):
        k = int(k)
        return 3 if k < 3 else (k if k % 2 else k + 1)

    se_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(close_ksize), odd(close_ksize)))
    se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(open_ksize),  odd(open_ksize)))
    se_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(ridge_dilate_ksize), odd(ridge_dilate_ksize)))

    for p in range(int(p_min), int(p_max) + 1):
        thr = float(np.percentile(gray_u8, float(p)))
        mask = (gray_u8 <= thr).astype(np.uint8) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_c)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  se_o)

        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        if n <= 1:
            continue

        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ring = (labels == k).astype(np.uint8)

        dt = cv2.distanceTransform(ring, cv2.DIST_L2, 5)
        dt_dil = cv2.dilate(dt, se_r)
        ridge = (dt > 0) & (dt == dt_dil)

        vals = dt[ridge]
        if vals.size == 0:
            continue

        ridge &= (dt >= np.percentile(vals, float(ridge_dt_percentile)))
        if int(np.count_nonzero(ridge)) >= int(ridge_min_points):
            return int(p)

    # fallback if nothing worked
    return int((p_min + p_max) // 2)
def center_crop_ring_detection_rgb(
    img: np.ndarray,
    crop_size: int = 2048,
    #p_dark: float = 8,
    close_ksize: int = 13,
    open_ksize: int = 9,
    ridge_dilate_ksize: int = 9,
    ridge_dt_percentile: float = 70,
) -> tuple[np.ndarray, int, int]:
    """
    Detect well center using dark-ring ridge method, then center-crop.

    Returns:
        crop  : cropped RGB image (crop_size x crop_size x 3)
        x0, y0: top-left crop corner in original image coordinates
    """

    # ------------------------------
    # checks
    # ------------------------------
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB (H,W,3). Got {img.shape}")

    H, W, _ = img.shape
    if crop_size > H or crop_size > W:
        raise ValueError(f"crop_size={crop_size} bigger than image size {(H,W)}")

    # ------------------------------
    # grayscale for dark mask
    # ------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # uint8
    p_dark = auto_p_dark_minimal(gray_u8)   
    thr = float(np.percentile(gray, float(p_dark)))
    mask = (gray <= thr).astype(np.uint8) * 255

    # ------------------------------
    # minimal morph cleanup
    # ------------------------------
    def odd(k):
        k = int(k)
        return 3 if k < 3 else (k if k % 2 else k + 1)

    se_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(close_ksize), odd(close_ksize)))
    se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(open_ksize),  odd(open_ksize)))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_c)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  se_o)

    # ------------------------------
    # largest connected component (ring)
    # ------------------------------
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    if n <= 1:
        raise RuntimeError("No ring component found. Increase p_dark or morph sizes.")

    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ring = (labels == k).astype(np.uint8)

    # ------------------------------
    # ridge from distance transform
    # ------------------------------
    dt = cv2.distanceTransform(ring, cv2.DIST_L2, 5)

    k_d = odd(ridge_dilate_ksize)
    dt_dil = cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_d, k_d)))
    ridge = (dt > 0) & (dt == dt_dil)

    vals = dt[ridge]
    if vals.size == 0:
        raise RuntimeError("No ridge points found.")

    ridge &= (dt >= np.percentile(vals, float(ridge_dt_percentile)))

    ys, xs = np.nonzero(ridge)
    if xs.size < 100:
        raise RuntimeError("Too few ridge points for stable center.")

    # ------------------------------
    # least-squares circle fit → center
    # ------------------------------
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x*x + y*y

    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, _ = c

    cx = float(cx)
    cy = float(cy)

    # ------------------------------
    # crop around detected center
    # ------------------------------
    half = crop_size // 2

    x0 = int(round(cx - half))
    y0 = int(round(cy - half))

    x0 = max(0, min(x0, W - crop_size))
    y0 = max(0, min(y0, H - crop_size))

    crop = img[y0:y0 + crop_size, x0:x0 + crop_size, :]

    return crop, x0, y0

# ============================================================
# Mask filtering / rendering utilities
# ============================================================
def filter_masks_by_area(
    masks: List[Dict[str, Any]],
    lower_area: int,
    upper_area: int,
) -> List[Dict[str, Any]]:
    out = []
    for m in masks:
        area = int(m.get("area", 0))
        if lower_area <= area <= upper_area:
            out.append(m)
    return out

def touches_corner(seg: np.ndarray) -> bool:
    """
    True if the mask touches BOTH a vertical border (left/right)
    AND a horizontal border (top/bottom) => corner-touching.
    """
    h, w = seg.shape[:2]
    left   = seg[:, 0].any()
    right  = seg[:, w - 1].any()
    top    = seg[0, :].any()
    bottom = seg[h - 1, :].any()

    touches_x = left or right
    touches_y = top or bottom
    return touches_x and touches_y

def masks_to_label_image(masks: List[Dict[str, Any]], h: int, w: int) -> np.ndarray:
    """
    Build a label image where each mask gets a unique integer ID (1..N).
    """
    lab = np.zeros((h, w), dtype=np.int32)
    for k, m in enumerate(masks, start=1):
        seg = m["segmentation"]
        lab[seg] = k
    return lab


def draw_overlay(rgb: np.ndarray, masks: List[Dict[str, Any]], alpha: float = 0.45) -> np.ndarray:
    """
    Overlay instance masks with deterministic random colors.
    """
    assert_uint8_rgb(rgb, name="overlay_input_rgb")
    overlay = rgb.copy()
    rng = np.random.default_rng(0)
    for m in masks:
        seg = m["segmentation"]
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        overlay[seg] = (overlay[seg] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay

def fishiness_score(seg_bool, debug=False):

    # clean mask
    m = (seg_bool.astype(np.uint8) * 255)
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    )
    m = (m > 0)

    # skeleton
    sk = skeletonize(m)
    L = int(sk.sum())

    if L == 0:
        return -1e9

    # thickness variation
    dt = distance_transform_edt(m)
    r = dt[sk]

    if r.size < 10:
        return -1e9

    width_var = float(np.std(r) / (np.mean(r) + 1e-6))

    score = L * (1.0 + 3.0 * width_var)

    if debug:
        print(f"  skel_len = {L}")
        print(f"  width_var = {width_var:.3f}")
        print(f"  score = {score:.2f}")

    return score

# ============================================================
# SAM run + save
# ============================================================
@dataclass
class SamRunConfig:
    lower_area: int = 50000
    upper_area: int = 200000
    save_overlay: bool = True
    save_label_image: bool = True
    save_individual_masks: bool = True

def touches_border(seg: np.ndarray) -> bool:
    """
    Returns True if the binary mask touches any image border.
    """
    return (
        seg[0, :].any() or
        seg[-1, :].any() or
        seg[:, 0].any() or
        seg[:, -1].any()
    )

def save_survivor_scores(
    clean_masks: List[Dict[str, Any]],
    rgb: np.ndarray,
    out_dir: Union[str, Path],
    base_name: str,
) -> None:
    """Write fishiness scores + labeled overlays for post-filter survivor masks."""
    d = Path(out_dir) / "candidate_scores" / base_name
    d.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, m in enumerate(clean_masks, start=1):
        seg = m["segmentation"]
        area = int(m.get("area", int(seg.astype(bool).sum())))
        score = float(fishiness_score(seg))
        predicted_iou = float(m.get("predicted_iou", float("nan")))
        stability_score = float(m.get("stability_score", float("nan")))
        rows.append((i, area, score, predicted_iou, stability_score))

        overlay = rgb.copy()
        overlay[seg.astype(bool)] = [0, 255, 0]
        cv2.putText(
            overlay, f"#{i} F={score:.0f} A={area} IoU={predicted_iou:.2f}", (6, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )
        cv2.imwrite(
            str(d / f"survivor_{i:03d}_area{area}_F{int(round(score))}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    with open(d / "survivors.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cand_id", "area", "fishiness", "predicted_iou", "stability_score"])
        writer.writerows(rows)

# ============================================================
# Mask selectors (dark-eye / arena geometry)
# ============================================================
DARK5_REJECT = 10.0       # reject mask if mean of darkest 5% in-mask gray px >= this (no dark eye/body)
ARENA_RADIUS_PAD = 15.0   # px grown onto the arena radius so a head/tail poking slightly past the wall still counts inside
ARENA_TOUCH_BAND = 3.0    # px: a mask "touches" the wall if it reaches within this of the (padded) radius

def dark5_mean(gray: np.ndarray, seg: np.ndarray) -> float:
    """Mean of the darkest 5% of in-mask grayscale pixels (NaN if mask empty)."""
    seg = seg.astype(bool)
    v = gray[seg].astype(np.float32)
    if v.size == 0:
        return float("nan")
    return float(v[v <= np.percentile(v, 5)].mean())

def _iou_key(m: Dict[str, Any]) -> float:
    v = float(m.get("predicted_iou", float("nan")))
    return v if np.isfinite(v) else -1.0

def arena_inside_metrics(seg: np.ndarray, cx: float, cy: float, radius: float) -> Tuple[bool, int]:
    """(inside_not_touching, area_inside) for one mask vs the padded arena circle."""
    seg = seg.astype(bool)
    if not seg.any():
        return False, 0
    ar = radius + ARENA_RADIUS_PAD
    ys, xs = np.nonzero(seg)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    area_inside = int((dist <= ar).sum())
    inside = bool(dist.max() <= ar - ARENA_TOUCH_BAND)
    return inside, area_inside

def select_hier_darkeye(clean_masks: List[Dict[str, Any]], gray: np.ndarray) -> Dict[str, Any]:
    """L-bend selector: keep masks with a dark eye (dark5_mean < DARK5_REJECT), then
    take the highest SAM predicted_iou. Falls back to all masks if none qualify."""
    eligible = [m for m in clean_masks if dark5_mean(gray, m["segmentation"]) < DARK5_REJECT]
    pool = eligible if eligible else clean_masks
    return max(pool, key=_iou_key)

def select_arena_darkeye(
    clean_masks: List[Dict[str, Any]],
    gray: np.ndarray,
    arena_circle: Optional[Tuple[float, float, float]],
) -> Dict[str, Any]:
    """Heartbeat selector: prefer masks fully inside the arena circle, then masks with
    a dark eye, then highest SAM predicted_iou (tie -> larger in-arena area)."""
    area_inside: Dict[int, int] = {}
    if arena_circle is not None:
        cx, cy, r = arena_circle
        inside_masks = []
        for m in clean_masks:
            ins, ain = arena_inside_metrics(m["segmentation"], cx, cy, r)
            area_inside[id(m)] = ain
            if ins:
                inside_masks.append(m)
        base_pool = inside_masks if inside_masks else clean_masks
    else:
        base_pool = clean_masks
    eligible = [m for m in base_pool if dark5_mean(gray, m["segmentation"]) < DARK5_REJECT]
    pool = eligible if eligible else base_pool
    return max(pool, key=lambda m: (_iou_key(m), area_inside.get(id(m), 0)))

def run_sam_on_rgb(
    mask_generator: SamAutomaticMaskGenerator,
    rgb: np.ndarray,
    cfg: SamRunConfig,
    out_dir: Path,
    base_name: str,
    circle_mask: np.ndarray | None = None,   # add this
    select_by: str = "fishiness",   # "fishiness" (default), "area", or "hier_darkeye"
    raw_rgb: np.ndarray | None = None,   # non arena-blacked crop, used for the dark-eye test
) -> Dict[str, Any]:
    """
    Run SAM on an RGB uint8 image and return results dict.
    """
    assert_uint8_rgb(rgb, name="sam_input_rgb")
    h, w = rgb.shape[:2]

    masks = mask_generator.generate(rgb)

    # area filter
    masks_f = filter_masks_by_area(masks,cfg.lower_area,cfg.upper_area)
    # -------------------------------
    # SAVE ALL AREA-FILTERED MASKS (PER WELL)
    # -------------------------------
    debug_root = Path(out_dir) / "debug_masks_area_filtered"
    well_dir = debug_root / base_name
    well_dir.mkdir(parents=True, exist_ok=True)

    for i, m in enumerate(masks_f, start=1):
        seg = m["segmentation"].astype(np.uint8) * 255

        cv2.imwrite(str(well_dir / f"mask_{i:03d}.png"), seg)

        overlay = rgb.copy()
        overlay[seg > 0] = [0, 255, 0]

        cv2.imwrite(
            str(well_dir / f"overlay_{i:03d}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

    # NEW: discard masks touching image borders
    # masks_f = [
    #     m for m in masks_f
    #     if not touches_border(m["segmentation"])
    # ]
    #print(f"    ➡️  {len(masks)} masks -> {len(masks_f)} after area filtering")
    # if len(masks_f) > 1: #cases where only one mask sometime touch the boarder
    #     masks_f = [
    #         m for m in masks_f
    #         if not touches_border(m["segmentation"])
    #     ]
 
    if len(masks_f) >= 1:

        clean_masks = [
            m for m in masks_f
            if not reject_edge_or_ring_mask(
                m["segmentation"],
                circle_mask=circle_mask,
                band_px=35,
                max_boundary_frac=0.30,
            )
        ]

        if len(clean_masks) > 0:
            save_survivor_scores(clean_masks, rgb, out_dir, base_name)
            if select_by == "area":
                best = max(clean_masks, key=lambda m: int(m.get("area", int(m["segmentation"].astype(bool).sum()))))
            elif select_by == "fishiness":
                best = max(clean_masks, key=lambda m: fishiness_score(m["segmentation"]))
            else:  # "hier_darkeye": dark-eye filter -> max predicted_iou (L-bend)
                # dark5 must read the RAW crop so masks spilling past the wall don't
                # pick up the arena-blacked 0 pixels and look falsely dark.
                gray = cv2.cvtColor(raw_rgb if raw_rgb is not None else rgb, cv2.COLOR_RGB2GRAY)
                best = select_hier_darkeye(clean_masks, gray)
            masks_f = [best]
        else:
            masks_f = []




    result: Dict[str, Any] = {"masks": masks_f}

    if cfg.save_label_image:
        result["label_image"] = masks_to_label_image(masks_f, h, w)

    if cfg.save_overlay:
        result["overlay"] = draw_overlay(rgb, masks_f)

    return result

def run_sam_on_rgb_hb(
    mask_generator: SamAutomaticMaskGenerator,
    rgb: np.ndarray,
    cfg: SamRunConfig,
    out_dir: Path | None = None,
    base_name: str = "sam_debug",
    circle_mask: np.ndarray | None = None,
    arena_circle: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Run SAM on an RGB uint8 image and return results dict.

    out_dir and base_name are optional so this function also works with:
        run_sam_on_rgb_fn(mask_generator, rgb_crop, cfg)
    """
    assert_uint8_rgb(rgb, name="sam_input_rgb")
    h, w = rgb.shape[:2]

    masks = mask_generator.generate(rgb)

    # area filter
    masks_f = filter_masks_by_area(
        masks,
        cfg.lower_area,
        cfg.upper_area,
    )

    # -------------------------------
    # SAVE ALL AREA-FILTERED MASKS ONLY IF out_dir IS PROVIDED
    # -------------------------------
    if out_dir is not None:
        debug_root = Path(out_dir) / "debug_masks_area_filtered"
        well_dir = debug_root / base_name
        well_dir.mkdir(parents=True, exist_ok=True)

        for i, m in enumerate(masks_f, start=1):
            seg = m["segmentation"].astype(np.uint8) * 255

            cv2.imwrite(str(well_dir / f"mask_{i:03d}.png"), seg)

            overlay = rgb.copy()
            overlay[seg > 0] = [0, 255, 0]

            cv2.imwrite(
                str(well_dir / f"overlay_{i:03d}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    # -------------------------------
    # Select best non-border / non-ring mask
    # -------------------------------
    if len(masks_f) >= 1:
        clean_masks = [
            m for m in masks_f
            if not reject_edge_or_ring_mask(
                m["segmentation"],
                circle_mask=circle_mask,
                band_px=35,
                max_boundary_frac=0.30,
            )
        ]

        if len(clean_masks) > 0:
            if out_dir is not None:
                save_survivor_scores(clean_masks, rgb, out_dir, base_name)
            # arena-geometry + dark-eye + max predicted_iou (rgb is the raw brightfield crop)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            best = select_arena_darkeye(clean_masks, gray, arena_circle)
            masks_f = [best]
        else:
            masks_f = []

    result: Dict[str, Any] = {"masks": masks_f}

    if cfg.save_label_image:
        result["label_image"] = masks_to_label_image(masks_f, h, w)

    if cfg.save_overlay:
        result["overlay"] = draw_overlay(rgb, masks_f)

    return result
def save_results(
    out_dir: Union[str, Path],
    base_name: str,
    frame_idx: int,
    rgb: np.ndarray,
    result: Dict[str, Any],
    cfg: SamRunConfig,
) -> None:
    """
    Save:
    - overlay PNG
    - labels NPY
    - individual instance masks PNG
    - input frame PNG (debug)
    """
    out_dir = Path(out_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (out_dir / "png").mkdir(parents=True, exist_ok=True)

    # Save overlay
    if cfg.save_overlay and "overlay" in result:
        ov = result["overlay"]
        cv2.imwrite(
            str(out_dir / "overlays" / f"{base_name}_f{frame_idx:04d}_overlay.png"),
            cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
        )

    # Save label image
    if cfg.save_label_image and "label_image" in result:
        lab = result["label_image"].astype(np.int32)
        np.save(out_dir / "masks" / f"{base_name}_f{frame_idx:04d}_labels.npy", lab)

    # Save individual masks as PNG (0/255)
    if cfg.save_individual_masks:
        masks = result["masks"]
        inst_dir = out_dir / "masks" / f"{base_name}_f{frame_idx:04d}_instances"
        inst_dir.mkdir(parents=True, exist_ok=True)

        for i, m in enumerate(masks, start=1):
            seg = m["segmentation"].astype(np.uint8) * 255
            cv2.imwrite(str(inst_dir / f"mask_{i:03d}.png"), seg)

    # Save the input frame too (useful for debugging / provenance)
    cv2.imwrite(
        str(out_dir / "png" / f"{base_name}_f{frame_idx:04d}_frame.png"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )


def make_plate_overlay_montage(
    png_dir: Path,
    frame_idx: int = 0,
    rows: str = "ABCDEFGH",
    cols: list[int] | None = None,
    out_name: str | None = None,
) -> Path:
    """
    Concatenate 96 well overlay PNGs into one plate montage.
    """

    if cols is None:
        cols = list(range(1, 13))

    # auto-generate montage filename using plate/output directory name
    if out_name is None:
        plate_name = png_dir.parent.name
        out_name = f"{plate_name}_overlay_montage.png"

    tile_rows = []

    for r in rows:
        row_imgs = []

        for c in cols:
            fname = f"well_{r}{c}_f{frame_idx:04d}_overlay.png"
            fpath = png_dir / fname
            print(fpath)

            if not fpath.exists():
                raise FileNotFoundError(f"Missing overlay: {fname}")

            img = cv2.imread(str(fpath))
            if img is None:
                raise RuntimeError(f"Failed to read: {fname}")

            row_imgs.append(img)

        row_concat = np.hstack(row_imgs)
        tile_rows.append(row_concat)

    plate_img = np.vstack(tile_rows)

    # keep montage inside overlays folder
    out_path = png_dir / out_name
    cv2.imwrite(str(out_path), plate_img)

    return out_path