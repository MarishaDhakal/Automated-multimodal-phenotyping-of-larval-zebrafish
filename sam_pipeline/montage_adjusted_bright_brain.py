#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# USER INPUT
# ============================================================
# Keep plate-specific calibration data beside this script.
PLATE_VALUES_CSV = Path(__file__).resolve().with_name("plate_values.csv")


# ============================================================
# MONTAGE FUNCTIONS
# ============================================================
def make_montages_for_folder(
    input_dir: Path,
    out_dir: Path,
    suffix: str,
    output_prefix: str,
    thumb_size=(220, 220),
):
    """
    Make 96-well montage PNGs for each frame number found in input_dir.

    Expected filenames:
        well_A1_f0000_overlay.png
        well_A1_f0000_frame.png
    """

    pattern = re.compile(
        rf"well_([A-H])(\d{{1,2}})_f(\d+)_{suffix}\.png$"
    )

    rows_labels = list("ABCDEFGH")
    cols_labels = list(range(1, 13))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    label_color = (255, 255, 255)

    border = 2

    tile_h = thumb_size[1] + 2 * border
    tile_w = thumb_size[0] + 2 * border

    plate_h = len(rows_labels) * tile_h
    plate_w = len(cols_labels) * tile_w

    files = list(input_dir.glob("*.png"))
    print(f"\nFound {len(files)} PNG files in {input_dir}")

    files_by_frame = {}

    for path in files:
        m = pattern.search(path.name)
        if not m:
            continue

        row_letter = m.group(1)
        col_number = int(m.group(2))
        frame_number = int(m.group(3))

        files_by_frame.setdefault(frame_number, []).append(
            (row_letter, col_number, path)
        )

    frame_list = sorted(files_by_frame.keys())
    print(f"{suffix} frames found: {frame_list}")

    if not frame_list:
        print(f"[WARN] No matching {suffix} PNG files found in {input_dir}")
        return

    for frame_number in frame_list:
        plate = np.zeros((plate_h, plate_w, 3), dtype=np.uint8)

        # --------------------------------------------------------
        # First create blank placeholders for all 96 wells
        # --------------------------------------------------------
        for row_letter in rows_labels:
            for col_number in cols_labels:
                r_i = rows_labels.index(row_letter)
                c_i = cols_labels.index(col_number)

                tile = np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255

                tile[
                    border:border + thumb_size[1],
                    border:border + thumb_size[0],
                ] = 0

                label = f"{row_letter}{col_number}  f{frame_number}"

                cv2.putText(
                    tile,
                    label,
                    (border + 8, border + 18),
                    font,
                    font_scale,
                    label_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

                y = r_i * tile_h
                x = c_i * tile_w

                plate[y:y + tile_h, x:x + tile_w] = tile

        # --------------------------------------------------------
        # Then overwrite available wells with actual images
        # --------------------------------------------------------
        for row_letter, col_number, path in files_by_frame[frame_number]:
            if row_letter not in rows_labels:
                continue

            if col_number not in cols_labels:
                continue

            r_i = rows_labels.index(row_letter)
            c_i = cols_labels.index(col_number)

            img = cv2.imread(str(path))

            if img is None:
                print(f"[WARN] Could not read image: {path}")
                continue

            img = cv2.resize(img, thumb_size)

            tile = np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255

            tile[
                border:border + thumb_size[1],
                border:border + thumb_size[0],
            ] = img

            label = f"{row_letter}{col_number}  f{frame_number}"

            cv2.putText(
                tile,
                label,
                (border + 8, border + 18),
                font,
                font_scale,
                label_color,
                font_thickness,
                cv2.LINE_AA,
            )

            y = r_i * tile_h
            x = c_i * tile_w

            plate[y:y + tile_h, x:x + tile_w] = tile

        out_path = out_dir / f"{output_prefix}_montage_f{frame_number:04d}.png"

        ok = cv2.imwrite(str(out_path), plate)

        if ok:
            print(f"Saved -> {out_path}")
        else:
            print(f"[WARN] Failed saving -> {out_path}")


# ============================================================
# SUMMARY FUNCTIONS
# ============================================================
def classify_bright_brain_from_top10_green(row) -> str:
    """
    Automatic classification based on crop_size and adjusted top-10 green mean.

    crop_size = 1048:
        bright_brain if fluoro_top10_green_mean_adjusted >= 95

    crop_size = 2048:
        bright_brain if fluoro_top10_green_mean_adjusted >= 115
    """

    crop_size = row.get("crop_size", np.nan)
    adjusted = row.get("fluoro_top10_green_mean_adjusted", np.nan)

    if pd.isna(crop_size) or pd.isna(adjusted):
        return "unknown"

    crop_size = int(round(float(crop_size)))
    adjusted = float(adjusted)

    if crop_size == 1048:
        threshold = 95.0
    elif crop_size == 2048:
        threshold = 115.0
    else:
        return "unknown_crop_size"

    if adjusted >= threshold:
        return "bright_brain"

    return "no_bright_brain"


def classification_rule_from_crop_size(crop_size) -> str:
    if pd.isna(crop_size):
        return "unknown crop_size"

    crop_size = int(round(float(crop_size)))

    if crop_size == 1048:
        return "crop_size=1048: fluoro_top10_green_mean_adjusted >= 95"

    if crop_size == 2048:
        return "crop_size=2048: fluoro_top10_green_mean_adjusted >= 115"

    return f"unknown crop_size={crop_size}"


def well_sort_key(well):
    try:
        well = str(well).upper().strip()
        row = well[0]
        col = int(well[1:])
        return (ord(row) - ord("A"), col)
    except Exception:
        return (999, 999)


def load_plate_values_csv(csv_path: Path) -> dict[str, float]:
    """Load plate_values.csv in A-H x 1-12 matrix format into {well: factor}."""
    df = pd.read_csv(csv_path)

    if "row" not in df.columns:
        raise ValueError(f"plate values CSV must contain a 'row' column: {csv_path}")

    factors = {}
    for _, r in df.iterrows():
        row_letter = str(r["row"]).strip().upper()
        if row_letter not in "ABCDEFGH":
            continue

        for col in range(1, 13):
            col_name = str(col)
            if col_name not in df.columns:
                raise ValueError(f"plate values CSV is missing column {col_name}: {csv_path}")

            value = pd.to_numeric(r[col_name], errors="coerce")
            if pd.notna(value):
                factors[f"{row_letter}{col}"] = float(value)

    return factors


def make_merged_summary_excel(
    root_dir: Path,
    out_dir: Path,
    output_filename: str = "merged_summary.xlsx",
    plate_values_csv: Path = PLATE_VALUES_CSV,
):
    """
    Find all *_gcamp_metrics_5frames_with_slopes.csv files,
    merge them, summarize by well, classify bright brain,
    and save Excel.
    """

    csv_files = sorted(
        root_dir.rglob("*_gcamp_metrics_5frames_with_slopes.csv")
    )

    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f}")

    if len(csv_files) == 0:
        print("[WARN] No CSV files found. Skipping Excel summary.")
        return

    dataframes = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["source_csv"] = str(csv_file)
            print(f"Read {csv_file}: {len(df)} rows")
            dataframes.append(df)

        except Exception as e:
            print(f"[WARN] Error reading {csv_file}: {e}")

    if not dataframes:
        print("[WARN] No dataframes were successfully read. Skipping Excel summary.")
        return

    merged_df = pd.concat(dataframes, ignore_index=True)

    print(f"\nMerged dataframe: {len(merged_df)} total rows")
    print(f"Columns: {list(merged_df.columns)}")

    if "well" not in merged_df.columns:
        raise ValueError("Merged dataframe does not contain a 'well' column.")

    # ------------------------------------------------------------
    # Convert only columns needed for this summary
    # ------------------------------------------------------------
    numeric_cols = [
        "top10_green",
        "crop_size",
        "mask_area",
        "n_points",
    ]

    for c in numeric_cols:
        if c in merged_df.columns:
            merged_df[c] = pd.to_numeric(merged_df[c], errors="coerce")

    # ------------------------------------------------------------
    # Summary by well
    # ------------------------------------------------------------
    agg_dict = {}

    # Main metric you want
    agg_dict["top10_green_mean"] = ("top10_green", "mean")
    agg_dict["top10_green_std"] = ("top10_green", "std")

    # Number of usable points from your original script
    if "n_points" in merged_df.columns:
        agg_dict["n_points"] = ("n_points", "first")

    # Crop size for automatic classification
    if "crop_size" in merged_df.columns:
        agg_dict["crop_size"] = ("crop_size", "first")

    # Total rows found for that well
    agg_dict["n_rows"] = ("well", "size")

    # Number of valid masks
    if "mask_area" in merged_df.columns:
        agg_dict["n_valid_masks"] = (
            "mask_area",
            lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum()),
        )

    # Combine notes
    if "note" in merged_df.columns:
        agg_dict["note"] = (
            "note",
            lambda x: "; ".join(
                sorted(
                    set(
                        str(v)
                        for v in x.dropna()
                        if str(v).strip() != ""
                    )
                )
            ),
        )

    summary = merged_df.groupby("well").agg(**agg_dict).reset_index()

    # ------------------------------------------------------------
    # Clean crop_size as integer
    # ------------------------------------------------------------
    if "crop_size" in summary.columns:
        summary["crop_size"] = summary["crop_size"].round().astype("Int64")

    # ------------------------------------------------------------
    # Apply per-well fluorescence adjustment from plate_values.csv
    # adjusted = fluoro_top10_green_mean / plate factor
    # ------------------------------------------------------------
    plate_values_csv = plate_values_csv.resolve()
    if not plate_values_csv.exists():
        raise FileNotFoundError(f"plate_values.csv not found: {plate_values_csv}")

    plate_factor_map = load_plate_values_csv(plate_values_csv)
    summary["plate_value_factor"] = summary["well"].map(plate_factor_map)

    # Use the requested source column when present.
    # For this script's current naming, top10_green_mean is the equivalent fallback.
    if "fluoro_top10_green_mean" in summary.columns:
        source_col = "fluoro_top10_green_mean"
    elif "top10_green_mean" in summary.columns:
        source_col = "top10_green_mean"
        summary["fluoro_top10_green_mean"] = summary[source_col]
    else:
        raise ValueError(
            "Summary does not contain 'fluoro_top10_green_mean' or 'top10_green_mean'."
        )

    summary["fluoro_top10_green_mean_adjusted"] = (
        pd.to_numeric(summary["fluoro_top10_green_mean"], errors="coerce")
        / pd.to_numeric(summary["plate_value_factor"], errors="coerce")
    )

    missing_factor = summary.loc[summary["plate_value_factor"].isna(), "well"].tolist()
    if missing_factor:
        print(f"[WARN] Missing plate-value factor for wells: {missing_factor}")

    print(f"Applied plate-value adjustment from: {plate_values_csv}")

    # ------------------------------------------------------------
    # Automatic bright-brain classification USING ADJUSTED VALUE
    # ------------------------------------------------------------
    summary["bright_brain_by_top10_green"] = summary.apply(
        classify_bright_brain_from_top10_green,
        axis=1,
    )

    summary["classification_rule"] = summary["crop_size"].apply(
        classification_rule_from_crop_size
    )

    # ------------------------------------------------------------
    # Sort wells A1-H12
    # ------------------------------------------------------------
    summary = summary.sort_values(
        by="well",
        key=lambda s: s.map(well_sort_key),
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Save Excel
    # ------------------------------------------------------------
    output_path = out_dir / output_filename

    # with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    #     merged_df.to_excel(
    #         writer,
    #         sheet_name="merged_all_frames",
    #         index=False,
    #     )

    #     summary.to_excel(
    #         writer,
    #         sheet_name="summary_by_well",
    #         index=False,
    #     )

    # print(f"\nSaved Excel summary -> {output_path}")
    # ------------------------------------------------------------
    # Save as CSV (no Excel dependency)
    # ------------------------------------------------------------
    merged_path = out_dir / "merged_all_frames.csv"
    summary_path = out_dir / "summary_by_well.csv"

    merged_df.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved merged data -> {merged_path}")
    print(f"Saved summary -> {summary_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Make 96-well montages and merged GCaMP summary."
    )

    ap.add_argument(
        "--root_dir",
        required=True,
        type=Path,
        help="Plate output root, e.g. OUT_ROOT/PLATE_NAME_per_well_fluorescence",
    )

    ap.add_argument(
        "--thumb_w",
        type=int,
        default=220,
        help="Thumbnail width.",
    )

    ap.add_argument(
        "--thumb_h",
        type=int,
        default=220,
        help="Thumbnail height.",
    )

    ap.add_argument(
        "--output_filename",
        default="merged_summary.xlsx",
        help="Excel output filename saved inside montages folder.",
    )

    args = ap.parse_args()

    root_dir = args.root_dir.resolve()

    overlay_dir = root_dir / "overlays"
    png_dir = root_dir / "png"

    out_dir = root_dir / "montages"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("ROOT_DIR    =", root_dir)
    print("OVERLAY_DIR =", overlay_dir)
    print("PNG_DIR     =", png_dir)
    print("OUT_DIR     =", out_dir)

    thumb_size = (args.thumb_w, args.thumb_h)

    # ------------------------------------------------------------
    # 1. Make overlay montages
    # ------------------------------------------------------------
    if overlay_dir.exists():
        make_montages_for_folder(
            input_dir=overlay_dir,
            out_dir=out_dir,
            suffix="overlay",
            output_prefix="plate_overlay",
            thumb_size=thumb_size,
        )
    else:
        print(f"[WARN] Missing overlay dir: {overlay_dir}")

    # ------------------------------------------------------------
    # 2. Make raw frame montages
    # ------------------------------------------------------------
    if png_dir.exists():
        make_montages_for_folder(
            input_dir=png_dir,
            out_dir=out_dir,
            suffix="frame",
            output_prefix="plate_frame",
            thumb_size=thumb_size,
        )
    else:
        print(f"[WARN] Missing png dir: {png_dir}")

    # ------------------------------------------------------------
    # 3. Merge all per-well CSVs and create Excel summary
    # ------------------------------------------------------------
    make_merged_summary_excel(
        root_dir=root_dir,
        out_dir=out_dir,
        output_filename=args.output_filename,
        plate_values_csv=PLATE_VALUES_CSV,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()