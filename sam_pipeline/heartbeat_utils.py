# heartbeat_utils.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import csv
import cv2
import numpy as np

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # safe for cluster/headless
import matplotlib.pyplot as plt

def _odd(k: int) -> int:
    k = int(k)
    return 3 if k < 3 else (k if k % 2 else k + 1)
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
    Pick the *smallest* percentile p in [p_min..p_max] that yields enough ridge points.
    Using the smallest p helps avoid including extra dark junk (shadows/fish) that
    higher p can add.
    """
    if gray_u8.dtype != np.uint8:
        raise ValueError("auto_p_dark_minimal expects gray_u8 uint8")

    se_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(close_ksize), _odd(close_ksize)))
    se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(open_ksize),  _odd(open_ksize)))
    se_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(ridge_dilate_ksize), _odd(ridge_dilate_ksize)))

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
    # auto p_dark search window
    p_min: int = 6,
    p_max: int = 14,
    # morph / ridge params
    close_ksize: int = 13,
    open_ksize: int = 9,
    ridge_dilate_ksize: int = 9,
    ridge_dt_percentile: float = 70,
    ridge_min_points: int = 100,
    # optional overrides / debug
    p_dark: Optional[int] = None,      # set int to force; None => auto
    return_debug: bool = False,
) -> Tuple[np.ndarray, int, int] | Tuple[np.ndarray, int, int, Dict[str, Any]]:
    """
    Detect well center using dark-ring ridge method, then crop around it.

    Args:
      img: RGB uint8 or RGB array (H,W,3). (If not uint8, percentile still works but cv2 ops assume uint8)
      crop_size: output crop size
      p_dark: force a percentile if you want; otherwise auto-selected per image
      return_debug: if True, returns (crop, x0, y0, debug_dict)

    Returns:
      crop: (crop_size,crop_size,3)
      x0,y0: top-left crop corner in original image coordinates
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB (H,W,3). Got {img.shape}")

    H, W, _ = img.shape
    if crop_size > H or crop_size > W:
        raise ValueError(f"crop_size={crop_size} bigger than image size {(H,W)}")

    # If img isn't uint8, convert for stable cv2 behavior
    if img.dtype != np.uint8:
        mx = float(img.max()) if img.size else 1.0
        mx = max(mx, 1.0)
        img_u8 = (img.astype(np.float32) / mx * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img_u8 = img

    gray_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)  # uint8

    # auto-select p_dark if not provided
    used_p = int(p_dark) if p_dark is not None else auto_p_dark_minimal(
        gray_u8,
        p_min=p_min, p_max=p_max,
        ridge_min_points=ridge_min_points,
        close_ksize=close_ksize,
        open_ksize=open_ksize,
        ridge_dilate_ksize=ridge_dilate_ksize,
        ridge_dt_percentile=ridge_dt_percentile,
    )

    thr = float(np.percentile(gray_u8, float(used_p)))
    mask = (gray_u8 <= thr).astype(np.uint8) * 255

    se_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(close_ksize), _odd(close_ksize)))
    se_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(open_ksize),  _odd(open_ksize)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_c)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  se_o)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        raise RuntimeError("No ring component found. Try wider p_min/p_max or adjust morph sizes.")

    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ring = (labels == k).astype(np.uint8)

    dt = cv2.distanceTransform(ring, cv2.DIST_L2, 5)
    se_r = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd(ridge_dilate_ksize), _odd(ridge_dilate_ksize)),
    )
    dt_dil = cv2.dilate(dt, se_r)
    ridge = (dt > 0) & (dt == dt_dil)

    vals = dt[ridge]
    if vals.size == 0:
        raise RuntimeError("No ridge points found. Increase close_ksize or increase p_max.")

    ridge &= (dt >= np.percentile(vals, float(ridge_dt_percentile)))

    ys, xs = np.nonzero(ridge)
    ridge_n = int(xs.size)
    if ridge_n < int(ridge_min_points):
        raise RuntimeError(f"Too few ridge points ({ridge_n}). Lower ridge_dt_percentile or increase p_max.")

    # least squares circle fit -> center
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x*x + y*y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, _ = c
    cx, cy = float(cx), float(cy)

    # crop around detected center
    half = crop_size // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x0 = max(0, min(x0, W - crop_size))
    y0 = max(0, min(y0, H - crop_size))

    crop = img_u8[y0:y0 + crop_size, x0:x0 + crop_size, :]

    if not return_debug:
        return crop, x0, y0

    debug = {
        "p_dark": used_p,
        "thr": thr,
        "ridge_points": ridge_n,
        "cx": cx,
        "cy": cy,
        "ring_area": int(stats[k, cv2.CC_STAT_AREA]),
    }
    return crop, x0, y0, debug








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


def ts_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_plate_wells(plate_dir: Path, pattern: str = "well_*.mp4") -> List[Tuple[str, Path]]:
    vids = sorted(plate_dir.glob(pattern))
    out = []
    for vp in vids:
        if vp.stem.startswith("well_"):
            well = vp.stem.split("well_", 1)[1]  # e.g. "A1"
            out.append((well, vp))
    return out


def get_video_frame_count(videopath: str) -> int:
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frame_bgr(videopath: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        raise RuntimeError(f"Video has 0 frames: {videopath}")

    if idx < 0:
        idx = n - 1
    idx = max(0, min(int(idx), n - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {idx} from {videopath}")
    return frame


# =========================
# SAM mask utilities
# =========================
def get_best_mask_from_sam_result(result: Dict) -> Optional[np.ndarray]:
    """
    Your run_sam_on_rgb returns {"masks": masks_f}
    Each m has m["segmentation"] bool.
    We pick the largest if multiple survive.
    """
    masks = result.get("masks", [])
    if not masks:
        return None
    best = max(masks, key=lambda m: int(m["segmentation"].sum()))
    return best["segmentation"].astype(bool)


def is_usable_mask(mask: Optional[np.ndarray], lower_area: int, upper_area: int) -> bool:
    if mask is None:
        return False
    a = int(mask.sum())
    return (a >= int(lower_area)) and (a <= int(upper_area))
def get_fixed_ring_crop_origin(
    videopath: str,
    crop_size: int = 1048,
) -> Tuple[int, int]:
    """
    Compute one fixed ring-detected crop origin from the first frame.
    This matches the notebook logic: detect crop once, reuse it.
    """
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from: {videopath}")

    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, xoff, yoff = center_crop_ring_detection_rgb(rgb_full, crop_size=crop_size)

    return int(xoff), int(yoff)

def find_valid_mask_by_search(
    videopath: str,
    mask_generator,
    run_sam_on_rgb_fn,
    cfg,
    start_frame: int,
    max_search: int = 60,
    direction: str = "forward",
    crop_size: int = 1048,
    fixed_xoff: Optional[int] = None,
    fixed_yoff: Optional[int] = None,
    cap=None,
) -> Tuple[
    Optional[np.ndarray],  # mask (crop coords)
    Optional[int],         # frame idx used
    Optional[np.ndarray], # full RGB frame
    Optional[int],         # x offset of crop
    Optional[int],         # y offset of crop
]:
    """
    Searches frames until a usable mask is found.

    SAM runs on center crop only.
    Motion + previews still use full frame.

    Returns:
      (mask_bool_crop, frame_idx, rgb_full, xoff, yoff)
    """

    owns_cap = cap is None
    if owns_cap:
        cap = cv2.VideoCapture(videopath)

    if cap is None or (not cap.isOpened()):
        raise RuntimeError(f"Could not open: {videopath}")

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if nframes <= 0:
        if owns_cap:
            cap.release()
        return None, None, None, None, None

    start_frame = max(0, min(int(start_frame), nframes - 1))
    max_search = max(1, int(max_search))

    if direction == "forward":
        indices = range(start_frame, min(start_frame + max_search, nframes))
    elif direction == "backward":
        indices = range(start_frame, max(start_frame - max_search, -1), -1)
    else:
        if owns_cap:
            cap.release()
        raise ValueError("direction must be 'forward' or 'backward'")

    lower_area = int(getattr(cfg, "lower_area"))
    upper_area = int(getattr(cfg, "upper_area"))

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # --- Full frame RGB (for preview + bbox mapping)
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Fixed crop if provided, otherwise detect per frame
        if fixed_xoff is not None and fixed_yoff is not None:
            xoff, yoff = int(fixed_xoff), int(fixed_yoff)
            rgb_crop = rgb_full[yoff:yoff + crop_size, xoff:xoff + crop_size]

            if rgb_crop.shape[0] != crop_size or rgb_crop.shape[1] != crop_size:
                continue
        else:
            # rgb_crop, xoff, yoff = center_crop_rgb(rgb_full, crop_size=crop_size)
            rgb_crop, xoff, yoff = center_crop_ring_detection_rgb(
                rgb_full,
                crop_size=crop_size,
            )

        # --- Run SAM on cropped frame
        result = run_sam_on_rgb_fn(mask_generator, rgb_crop, cfg)
        mask = get_best_mask_from_sam_result(result)

        if not is_usable_mask(mask, lower_area, upper_area):
            continue

        if owns_cap:
            cap.release()
        return mask, idx, rgb_full, xoff, yoff

    if owns_cap:
        cap.release()
    return None, None, None, None, None

# =========================
# Union bbox from two masks
# =========================
def union_bbox_from_masks(
    mask0: Optional[np.ndarray],
    mask1: Optional[np.ndarray],
    pad: int = 20,
    dilate_px: int = 10,
    min_size: int = 50,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """
    bbox = (x0,y0,x1,y1) inclusive pixel coords.
    union = bool mask
    """
    if mask0 is None and mask1 is None:
        return None, None

    if mask0 is None:
        union = mask1.copy()
    elif mask1 is None:
        union = mask0.copy()
    else:
        union = (mask0 | mask1)

    if dilate_px and dilate_px > 0:
        k = 2 * (dilate_px // 2) + 1
        union = cv2.dilate(
            union.astype(np.uint8),
            np.ones((k, k), np.uint8),
            iterations=1,
        ).astype(bool)

    ys, xs = np.where(union)
    if xs.size == 0:
        return None, union

    H, W = union.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)

    # enforce min size
    if (x1 - x0 + 1) < min_size:
        cx = (x0 + x1) // 2
        half = min_size // 2
        x0 = max(0, cx - half)
        x1 = min(W - 1, cx + half)

    if (y1 - y0 + 1) < min_size:
        cy = (y0 + y1) // 2
        half = min_size // 2
        y0 = max(0, cy - half)
        y1 = min(H - 1, cy + half)

    return (x0, y0, x1, y1), union


def save_union_mask_png(union_bool: np.ndarray, outpath: Path) -> None:
    img = (union_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(outpath), img)


def save_bbox_preview_rgb(rgb: np.ndarray, bbox: Tuple[int, int, int, int], outpath: Path) -> None:
    x0, y0, x1, y1 = bbox
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.imwrite(str(outpath), bgr)


# =========================
# Motion / heatmap / metrics
# =========================
def estimate_diff_thresh(
    videopath: str,
    bbox: Tuple[int, int, int, int],
    sample_frames: int = 60,
    k: float = 6.0,
    clamp_low: int = 2,
    clamp_high: int = 30,
) -> int:
    """
    Auto threshold from ROI diff distribution (median + k*MAD).
    Good for non-fluorescence too.
    """
    x0, y0, x1, y1 = bbox
    xs0, xs1 = x0, x1 + 1
    ys0, ys1 = y0, y1 + 1

    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        return 5

    ret, prev = cap.read()
    if not ret or prev is None:
        cap.release()
        return 5

    prev_roi = cv2.cvtColor(prev[ys0:ys1, xs0:xs1], cv2.COLOR_BGR2GRAY)

    diffs = []
    for _ in range(int(sample_frames)):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        roi = cv2.cvtColor(frame[ys0:ys1, xs0:xs1], cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(roi, prev_roi)
        diffs.append(d.reshape(-1))
        prev_roi = roi

    cap.release()
    if not diffs:
        return 5

    v = np.concatenate(diffs).astype(np.float32)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) + 1e-6)
    t = int(np.clip(med + k * mad, clamp_low, clamp_high))
    return t


def generate_motion_sum(
    videopath: str,
    bbox: Tuple[int, int, int, int],
    diff_thresh: int,
) -> Tuple[np.ndarray, int]:
    """
    Accumulate THRESH_TOZERO(absdiff) inside bbox.
    Returns (motion_sum_float32, frames_used)
    """
    x0, y0, x1, y1 = bbox
    xs0, xs1 = x0, x1 + 1
    ys0, ys1 = y0, y1 + 1

    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        cap.release()
        raise RuntimeError("Could not read first frame for motion analysis.")

    prev_roi = cv2.cvtColor(prev_frame[ys0:ys1, xs0:xs1], cv2.COLOR_BGR2GRAY)
    motion_sum = np.zeros(prev_roi.shape, dtype=np.float32)

    frames_used = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        curr_roi = cv2.cvtColor(frame[ys0:ys1, xs0:xs1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_roi, prev_roi)

        _, diff_tz = cv2.threshold(diff, int(diff_thresh), 255, cv2.THRESH_TOZERO)
        motion_sum += diff_tz.astype(np.float32)

        prev_roi = curr_roi
        frames_used += 1

    cap.release()
    return motion_sum, frames_used


def motion_sum_to_heatmap_bgr(motion_sum: np.ndarray) -> np.ndarray:
    heatmap_norm = cv2.normalize(motion_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_HOT)
    return heatmap_bgr


def compute_motion_metrics(motion_sum: np.ndarray) -> Dict[str, float]:
    total_motion = float(np.sum(motion_sum))
    peak_motion = float(np.max(motion_sum))
    mean_motion = float(np.mean(motion_sum))
    snr = float(peak_motion / (mean_motion + 1e-6))
    return {
        "motion_total": total_motion,
        "motion_peak": peak_motion,
        "motion_mean": mean_motion,
        "snr_peak_over_mean": snr,
    }


# =========================
# Results + CSV
# =========================
@dataclass
class WellResult:
    well: str
    videopath: str
    status: str
    error: str

    video_frame_count: int
    frame_a_used: int
    frame_b_used: int

    bbox_x0: int
    bbox_y0: int
    bbox_x1: int
    bbox_y1: int

    diff_thresh_used: int
    motion_frames_used: int
    motion_total: float
    motion_peak: float
    motion_mean: float
    snr_peak_over_mean: float

    union_mask_png: str
    heatmap_png: str
    bbox_preview_png: str


def to_row_dict(r: WellResult) -> Dict:
    return r.__dict__


# def write_results_csv(rows: List[Dict], csv_path: Path) -> None:
#     if not rows:
#         return
#     with open(csv_path, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         w.writeheader()
#         w.writerows(rows)


def bbox_add_offset(
    bbox: Tuple[int, int, int, int],
    xoff: int,
    yoff: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (x0 + xoff, y0 + yoff, x1 + xoff, y1 + yoff)
#--------------------------------------------------------------------


def smooth_1d(x: np.ndarray, win: int = 9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if win <= 1:
        return x
    win = int(win)
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, k, mode="same")


def connected_components_1d(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0])
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size - 1]
    return list(zip(starts, ends))  # inclusive


def save_motion_with_low_shading_png(
    motion: np.ndarray,
    fps: float,
    frame_idxs: np.ndarray,
    thr: float,
    smooth_win: int,
    outpath: Path,
    title: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Saves coarse motion plot with LOW shaded. Returns (motion_smooth, labels_low).
    """
    m = smooth_1d(motion, smooth_win)
    t = frame_idxs / float(fps)
    low = m < float(thr)

    plt.figure(figsize=(12, 4))
    plt.plot(t, m, linewidth=1)
    plt.axhline(float(thr), linestyle="--")
    plt.fill_between(t, m.min(), m.max(), where=low, alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Summed coarse motion (a.u.)")
    plt.title(title if title else f"Coarse motion (LOW shaded), thr={thr}")
    plt.tight_layout()
    plt.savefig(str(outpath), dpi=200, bbox_inches="tight")
    plt.close()

    return m, low

# heartbeat_utils.py (ADD)

def coarse_motion_trace(
    videopath: str,
    bbox: Tuple[int, int, int, int],
    downscale: int = 4,
    max_frames: Optional[int] = None,
) -> Dict[str, np.ndarray | float]:
    x0, y0, x1, y1 = bbox
    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame")

    prev = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    if downscale > 1:
        prev = cv2.resize(
            prev,
            (max(2, prev.shape[1] // downscale), max(2, prev.shape[0] // downscale)),
            interpolation=cv2.INTER_AREA,
        )

    motion = []
    frame_idxs = []
    f = 1
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)

        diff = cv2.absdiff(curr, prev).astype(np.float32)

        motion.append(float(diff.sum()))
        frame_idxs.append(f)

        prev = curr
        f += 1
        processed += 1
        if max_frames is not None and processed >= int(max_frames):
            break

    cap.release()
    return {
        "fps": float(fps),
        "frame_idxs": np.array(frame_idxs, dtype=np.int32),
        "motion": np.array(motion, dtype=np.float32),
    }

def analyze_fish_vitality_epoch(
    videopath, bbox, start_frame, end_frame,
    diff_thresh_val=5,
    top_frac=0.10,
):
    x0, y0, x1, y1 = bbox
    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read frame {start_frame}")

    roi0 = frame0[y0:y1, x0:x1]
    prev = cv2.cvtColor(roi0, cv2.COLOR_BGR2GRAY)

    motion_sum = np.zeros(prev.shape, dtype=np.float32)
    f = int(start_frame)
    n = 0

    while f < int(end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        f += 1

        roi = frame[y0:y1, x0:x1]
        curr = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(curr, prev)
        _, diff_t = cv2.threshold(diff, diff_thresh_val, 255, cv2.THRESH_TOZERO)
        diff_t = diff_t.astype(np.float32)

        motion_sum += diff_t
        prev = curr
        n += 1

    cap.release()

    # ---------- NEW: normalize by time (per frame-diff) ----------
    denom = max(n, 1)
    motion_mean = motion_sum / denom   # "motion per diff"

    # top-X% mean on the normalized heatmap
    vals = motion_mean.ravel()
    frac = min(max(float(top_frac), 0.0), 1.0)
    if vals.size == 0 or frac <= 0:
        top_mean = float("nan") if vals.size == 0 else 0.0
    else:
        k = max(1, int(frac * vals.size))
        top_vals = np.partition(vals, -k)[-k:]
        top_mean = float(top_vals.mean())

    seconds = n / float(fps)

    return {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "fps": float(fps),
        "n_diffs": int(n),
        "seconds": float(seconds),

        "roi_first": roi0,              # BGR
        "motion_sum": motion_sum,       # accumulated heatmap
        "motion_mean": motion_mean,     # NEW: per-frame heatmap (comparable!)

        # accumulated metrics (depend on epoch length)
        "total_motion": float(motion_sum.sum()),
        "peak_motion": float(motion_sum.max()),
        "avg_motion": float(motion_sum.mean()),

        # normalized metrics (comparable across epoch lengths)
        "total_motion_rate": float(motion_mean.sum()),     # per-diff sum over pixels
        "peak_motion_rate": float(motion_mean.max()),
        "avg_motion_rate": float(motion_mean.mean()),
        "top10_motion_rate": float(top_mean),
    }
def save_heatmap_png_from_float(
    heat_float: np.ndarray,
    outpath: Path,
    colormap: int = cv2.COLORMAP_HOT,
) -> None:
    h = cv2.normalize(heat_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.applyColorMap(h, colormap)
    cv2.imwrite(str(outpath), bgr)

def write_results_csv(rows: List[Dict], csv_path: Path) -> None:
    if not rows:
        return

    # Union of all keys across all rows
    fieldnames = sorted({k for r in rows for k in r.keys()})

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mask_crop_to_fullframe(
    videopath: str,
    mask_crop: np.ndarray,   # bool (crop_h, crop_w)
    xoff: int,
    yoff: int,
) -> np.ndarray:
    """
    Paste a crop-coordinate mask back into full-frame coordinates by using (xoff,yoff).
    Uses frame0 to determine full (H,W).
    """
    mask_crop = np.asarray(mask_crop, dtype=bool)

    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    ret, frame0 = cap.read()
    cap.release()
    if not ret or frame0 is None:
        raise RuntimeError("Could not read first frame")

    H, W = frame0.shape[:2]
    full = np.zeros((H, W), dtype=bool)

    ch, cw = mask_crop.shape[:2]
    x0, y0 = int(xoff), int(yoff)
    x1 = min(W, x0 + cw)
    y1 = min(H, y0 + ch)

    if x1 <= x0 or y1 <= y0:
        return full

    full[y0:y1, x0:x1] = mask_crop[: (y1 - y0), : (x1 - x0)]
    return full


def analyze_fish_vitality_full_mask(
    videopath,
    union_mask,                  # bool (H,W) same as frame size
    diff_thresh_val=4,
    top_frac=0.10,
    max_frames=None,             # e.g., 900 for 30s @30fps
    stride=1,
    flicker_correct=True,
):
    union_mask = np.asarray(union_mask, dtype=bool)

    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {videopath}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame")

    H, W = frame0.shape[:2]
    if union_mask.shape != (H, W):
        cap.release()
        raise ValueError(f"union_mask shape {union_mask.shape} != frame shape {(H, W)}")

    prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    motion_sum = np.zeros((H, W), dtype=np.float32)
    n = 0
    frames_read = 1

    while True:
        frame = None
        for _ in range(max(1, int(stride))):
            ret, frame = cap.read()
            if not ret:
                frame = None
                break
            frames_read += 1
            if max_frames is not None and frames_read >= int(max_frames):
                break

        if frame is None:
            break

        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr, prev).astype(np.float32)

        if flicker_correct:
            vals = diff[union_mask]
            if vals.size > 0:
                diff -= np.median(vals)
                diff[diff < 0] = 0.0

        diff[diff < float(diff_thresh_val)] = 0.0
        diff[~union_mask] = 0.0

        motion_sum += diff
        prev = curr
        n += 1

        if max_frames is not None and frames_read >= int(max_frames):
            break

    cap.release()

    # ✅ DEFINE motion_mean (this was missing)
    denom = max(n, 1)
    motion_mean = motion_sum / denom

    mask_area_px = int(np.count_nonzero(union_mask))
    area_denom = max(mask_area_px, 1)

    ms = motion_sum[union_mask]
    mm = motion_mean[union_mask]

    # time-normalized (per frame diff) inside fish
    total_motion_rate = float(mm.sum())
    avg_motion_rate_in_mask = float(mm.mean()) if mask_area_px > 0 else float("nan")
    peak_motion_rate_in_mask = float(mm.max()) if mask_area_px > 0 else float("nan")

    # pixel-normalized
    total_motion_per_pixel = float(ms.sum()) / area_denom
    total_motion_rate_per_pixel = total_motion_rate / area_denom  # per-pixel per-frame (best scalar)

    # top-X% mean on normalized heatmap (inside fish only)
    vals = mm.ravel()
    frac = min(max(float(top_frac), 0.0), 1.0)
    if vals.size == 0 or frac <= 0:
        top_mean = float("nan") if vals.size == 0 else 0.0
    else:
        k = max(1, int(frac * vals.size))
        top_vals = np.partition(vals, -k)[-k:]
        top_mean = float(top_vals.mean())

    seconds = n / float(fps)

    roi_first = frame0.copy()
    roi_first[~union_mask] = 0

    return {
        "fps": float(fps),
        "n_diffs": int(n),
        "seconds": float(seconds),

        "mask_area_px": int(mask_area_px),

        "roi_first": roi_first,
        "union_mask": union_mask,

        "motion_sum": motion_sum,
        "motion_mean": motion_mean,

        # accumulated metrics (depend on duration) — inside fish
        "total_motion": float(ms.sum()),
        "peak_motion": float(ms.max()) if mask_area_px > 0 else float("nan"),
        "avg_motion": float(ms.mean()) if mask_area_px > 0 else float("nan"),

        # normalized across frames (time) — inside fish
        "total_motion_rate": float(total_motion_rate),
        "peak_motion_rate": float(peak_motion_rate_in_mask),
        "avg_motion_rate": float(avg_motion_rate_in_mask),
        "top10_motion_rate": float(top_mean),

        # normalized across pixels (area) — inside fish
        "total_motion_per_pixel": float(total_motion_per_pixel),

        # normalized across frames + pixels — inside fish
        "total_motion_rate_per_pixel": float(total_motion_rate_per_pixel),
    }
 