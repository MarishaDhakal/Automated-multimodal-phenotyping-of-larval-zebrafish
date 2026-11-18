# ==========================================================
# make_heartbeat_montages.py
# Builds montage PNGs and collates heartbeat CSV summaries
# Saves all outputs into ROOT_DIR / "montages"
# ==========================================================

from pathlib import Path
import string
import argparse
import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ==========================================================
# Helper: load font safely
# ==========================================================
def load_font(size):
    """
    Try common fonts. If unavailable, fall back to PIL default font.
    """
    font_candidates = [
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass

    return ImageFont.load_default()


# ==========================================================
# Helper: sort wells naturally
# ==========================================================
def well_sort_key(w):
    """
    Sort wells as A1, A2, ..., A12, B1, ..., H12.
    """
    m = re.match(r"^([A-H])(\d{1,2})$", str(w).strip().upper())
    if m:
        return (m.group(1), int(m.group(2)))
    return ("Z", 999)


# ==========================================================
# Function: build montage for ONE image type
# ==========================================================
def build_montage(
    root_path,
    image_type,
    subdir,
    suffix,
    rows,
    cols,
    montage_dir,
    border_px=6,
    label_pad=6,
    border_color=(255, 255, 255),
    label_color=(0, 255, 0),
    inside_alpha_box=True,
    font_big=None,
    font_small=None,
):
    print(f"\nBuilding montage for: {image_type}")

    root_path = Path(root_path)
    montage_dir = Path(montage_dir)
    montage_dir.mkdir(parents=True, exist_ok=True)

    def get_well_folder(well):
        """
        Example root child:
            root_path / well_A5_longest_baseline_epoch_fallback_ica
        """
        candidates = [
            d for d in root_path.iterdir()
            if d.is_dir() and d.name.startswith(f"well_{well}_")
        ]

        return candidates[0] if candidates else None

    def get_image_path(well):
        """
        Handles two cases:

        Case 1:
        image is directly inside root_path

            root_path / well_A1_inside_minus_outside_t2_epochs.png

        Case 2:
        image is inside well folder / subdir

            root_path / well_A1_xxx / spatial_maps / well_A1_epoch_spatialmaps_all.png
        """
        filename = f"well_{well}{suffix}"

        if subdir is None:
            return root_path / filename

        well_folder = get_well_folder(well)

        if well_folder is None:
            return None

        return well_folder / subdir / filename

    # ----------------------------------------------------------
    # Find sample image to get tile size
    # ----------------------------------------------------------
    sample_found = False
    img_width = None
    img_height = None

    for row in rows:
        for col in cols:
            well = f"{row}{col}"
            sample_path = get_image_path(well)

            if sample_path is None:
                continue

            if sample_path.exists():
                with Image.open(sample_path) as sample_img:
                    sample_img = sample_img.convert("RGB")
                    img_width, img_height = sample_img.size

                sample_found = True
                print(f"Sample image found: {sample_path}")
                print(f"Image size: {img_width} x {img_height}")
                break

        if sample_found:
            break

    if not sample_found:
        print(f"No sample image found for: {image_type}")
        return None

    # ----------------------------------------------------------
    # Tile and montage canvas
    # ----------------------------------------------------------
    tile_w = img_width + 2 * border_px
    tile_h = img_height + 2 * border_px

    montage_w = tile_w * len(cols)
    montage_h = tile_h * len(rows)

    montage = Image.new("RGB", (montage_w, montage_h), (0, 0, 0))

    # ----------------------------------------------------------
    # Loop through wells
    # ----------------------------------------------------------
    n_found = 0
    n_missing = 0

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            well = f"{row}{col}"
            filepath = get_image_path(well)

            x0 = j * tile_w
            y0 = i * tile_h

            if filepath is None:
                print(f"Missing well folder for: {well}")
                n_missing += 1
                continue

            if not filepath.exists():
                print(f"Missing image: {filepath}")
                n_missing += 1
                continue

            # --------------------------------------------------
            # Load image
            # --------------------------------------------------
            with Image.open(filepath) as img:
                img = img.convert("RGB")

                # Resize only if needed to match sample size
                if img.size != (img_width, img_height):
                    print(
                        f"Resizing {filepath.name} "
                        f"from {img.size} to {(img_width, img_height)}"
                    )
                    img = img.resize((img_width, img_height))

                # --------------------------------------------------
                # Create bordered tile
                # --------------------------------------------------
                tile = Image.new("RGB", (tile_w, tile_h), border_color)
                tile.paste(img, (border_px, border_px))

            draw = ImageDraw.Draw(tile)

            # ======================================================
            # 1. Outer label on border
            # ======================================================
            outer_xy = (label_pad, label_pad)

            # Black outline
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text(
                    (outer_xy[0] + dx, outer_xy[1] + dy),
                    well,
                    font=font_big,
                    fill=(0, 0, 0),
                )

            draw.text(
                outer_xy,
                well,
                font=font_big,
                fill=label_color,
            )

            # ======================================================
            # 2. Inner label inside image
            # ======================================================
            inner_x = border_px + 10
            inner_y = border_px + img_height - 45

            if inside_alpha_box:
                box_w, box_h = 90, 42
                draw.rectangle(
                    [
                        inner_x - 5,
                        inner_y - 5,
                        inner_x + box_w,
                        inner_y + box_h,
                    ],
                    fill=(0, 0, 0),
                )

            # Black outline
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text(
                    (inner_x + dx, inner_y + dy),
                    well,
                    font=font_small,
                    fill=(0, 0, 0),
                )

            draw.text(
                (inner_x, inner_y),
                well,
                font=font_small,
                fill=label_color,
            )

            # ------------------------------------------------------
            # Paste tile into montage
            # ------------------------------------------------------
            montage.paste(tile, (x0, y0))
            n_found += 1

    # ----------------------------------------------------------
    # Save montage PNG inside montages folder
    # ----------------------------------------------------------
    output_name = f"montage{suffix}"
    output_path = montage_dir / output_name

    montage.save(output_path, dpi=(300, 300))

    print(f"Saved montage PNG: {output_path}")
    print(f"Images found: {n_found}")
    print(f"Images missing: {n_missing}")

    return output_path


# ==========================================================
# Function: collate heartbeat CSV summaries
# ==========================================================
def collate_heartbeat_data(root_path, montage_dir):
    """
    Collate all *_epoch_blob_summary.csv files and create:

      1. merged_fallback_ic_summary.csv
      2. merged_fallback_well_summary.csv

    Outputs are saved into ROOT_DIR / montages.

    Also attempts Excel export if openpyxl is installed.
    """
    root_path = Path(root_path)
    montage_dir = Path(montage_dir)
    montage_dir.mkdir(parents=True, exist_ok=True)

    print("\n==========================================================")
    print("Collating heartbeat, mobility, and corrected-fallback summaries")
    print("==========================================================")

    # ----------------------------------------------------------
    # Find all blob summary CSVs
    # ----------------------------------------------------------
    csv_files = sorted(root_path.rglob("*_epoch_blob_summary.csv"))

    print(f"Found {len(csv_files)} epoch blob summary CSV files.")

    if len(csv_files) == 0:
        print(
            "No *_epoch_blob_summary.csv files found. "
            "Continuing with plate_summary.csv so wells whose fallback "
            "was skipped can still be summarized."
        )

    # ----------------------------------------------------------
    # Read and merge
    # ----------------------------------------------------------
    dataframes = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if len(df) == 0:
                print(f"Skipping empty CSV: {csv_file}")
                continue

            print(f"Read {csv_file}: {len(df)} rows")
            dataframes.append(df)

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if len(dataframes) == 0:
        # A plate can legitimately have no fallback blob CSVs when all
        # relevant wells were already classified alive_by_slight_motion.
        merged_df = pd.DataFrame(columns=[
            "well",
            "reject_reason",
            "peak_signal_name",
            "n_peaks_real",
            "is_periodic",
            "best_candidate",
        ])
    else:
        merged_df = pd.concat(dataframes, ignore_index=True)

        if "well" not in merged_df.columns:
            raise RuntimeError(
                "Merged blob dataframe has no 'well' column."
            )

        merged_df["well"] = (
            merged_df["well"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        merged_df = merged_df.sort_values(
            by="well",
            key=lambda s: s.map(well_sort_key),
        ).reset_index(drop=True)

    # ----------------------------------------------------------
    # Save merged IC/blob-level summary
    # ----------------------------------------------------------
    ic_summary_csv = montage_dir / "merged_fallback_ic_summary.csv"
    merged_df.to_csv(ic_summary_csv, index=False)

    print(f"Saved merged IC/blob summary CSV: {ic_summary_csv}")
    print(f"Merged dataframe shape: {merged_df.shape}")

    # ----------------------------------------------------------
    # Prepare required columns for well-level summary
    # ----------------------------------------------------------
    if "reject_reason" not in merged_df.columns:
        merged_df["reject_reason"] = ""

    if "peak_signal_name" not in merged_df.columns:
        merged_df["peak_signal_name"] = ""

    if "n_peaks_real" not in merged_df.columns:
        merged_df["n_peaks_real"] = pd.NA

    if "is_periodic" not in merged_df.columns:
        merged_df["is_periodic"] = pd.NA

    if "best_candidate" not in merged_df.columns:
        merged_df["best_candidate"] = 0

    accepted_mask = (
        merged_df["reject_reason"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    # ----------------------------------------------------------
    # Well-level summarization
    # ----------------------------------------------------------
    def summarize_well(group):
        alive_rows = group[accepted_mask.loc[group.index]].copy()

        is_alive = len(alive_rows) > 0
        alive_dead = "alive" if is_alive else "dead"

        if not is_alive:
            type_of_hb = "dead"
        else:
            peak_names = (
                alive_rows["peak_signal_name"]
                .dropna()
                .astype(str)
                .str.strip()
            )

            unique_peak_names = sorted(set([x for x in peak_names if x != ""]))

            if "bandpassed" in unique_peak_names:
                type_of_hb = "strong"
            elif len(unique_peak_names) > 0 and set(unique_peak_names) == {"hilbert_envelope"}:
                type_of_hb = "weak"
            else:
                type_of_hb = "weak"

        # ------------------------------------------------------
        # Pull extra fields from best_candidate row only
        # ------------------------------------------------------
        best_rows = group[
            pd.to_numeric(group["best_candidate"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        ].copy()

        if len(best_rows) > 0:
            best_row = best_rows.iloc[0]

            best_n_peaks_real = best_row.get("n_peaks_real", pd.NA)
            best_peak_signal_name = best_row.get("peak_signal_name", pd.NA)
            best_is_periodic = best_row.get("is_periodic", pd.NA)
        else:
            best_n_peaks_real = pd.NA
            best_peak_signal_name = pd.NA
            best_is_periodic = pd.NA

        return pd.Series({
            "alive_dead": alive_dead,
            "type_of_hb": type_of_hb,
            "n_peaks_real": best_n_peaks_real,
            "peak_signal_name": best_peak_signal_name,
            "is_periodic": best_is_periodic,
        })

    if len(merged_df) > 0:
        summary_df = (
            merged_df.groupby("well", as_index=False)
            .apply(summarize_well)
            .reset_index(drop=True)
        )

        summary_df["assessed_by"] = "Heartbeat_detector"

    else:
        summary_df = pd.DataFrame(columns=[
            "well",
            "alive_dead",
            "type_of_hb",
            "n_peaks_real",
            "peak_signal_name",
            "is_periodic",
            "assessed_by",
        ])


    # ==========================================================
    # Merge mobility results into ALL wells
    # ==========================================================
    # A well can have an ICA/blob summary but still have no accepted
    # heartbeat candidate. In that case, a positive mobility call must
    # override the heartbeat-detector "dead" result.
    #
    # Priority:
    #   1. Accepted heartbeat -> alive / Heartbeat_detector
    #   2. Otherwise, mobility_alive_final == 1 -> alive / Mobility
    #   3. Otherwise retain heartbeat-dead or Unknown
    # ==========================================================

    plate_summary_path = root_path / "plate_summary.csv"

    if plate_summary_path.exists():
        try:
            plate_df = pd.read_csv(plate_summary_path)
        except Exception as e:
            print(f"ERROR reading plate summary {plate_summary_path}: {e}")
            plate_df = pd.DataFrame()
    else:
        print(f"plate_summary.csv not found: {plate_summary_path}")
        plate_df = pd.DataFrame()

    mobility_columns = [
        "well",

        # Motion-tier classification
        "max_pos_dev_from_baseline",
        "pos_dev_threshold",
        "strong_motion_peak_threshold",
        "slight_motion_peak_low",
        "slight_motion_peak_high",
        "min_slight_motion_peaks",
        "n_motion_peaks_total_ge_0p01",
        "n_strong_motion_peaks",
        "n_slight_motion_peaks",
        "mobility_alive_by_strong_motion",
        "mobility_alive_by_slight_motion",
        "mobility_alive_by_overlap",
        "mobility_alive_by_pos_dev",
        "mobility_alive_final",
        "mobility_motion_tier",
        "mobility_alive_call",
        "epoch_mask_overlap_frac",
        "epoch_mask_overlap_threshold",

        # Longest-epoch fallback status
        "used_longest_epoch_fallback",
        "longest_epoch_fallback_reason",
        "fallback_skipped_alive_by_slight_motion",

        # In-memory fallback motion correction
        "fallback_motion_correction_used",
        "fallback_masks_motion_corrected",
        "fallback_selected_correction_sign",
        "fallback_camera_motion_csv",
        "fallback_camera_motion_median_px",
        "fallback_camera_motion_p95_px",
        "fallback_camera_motion_max_px",
        "fallback_camera_motion_ecc_failed_n",
        "fallback_outside_p95_sign_neg1",
        "fallback_outside_p95_sign_pos1",

        # Fallback output/QC paths
        "epoch_union_mask_npy",
        "epoch_ica_summary_csv",
        "epoch_ica_time_pngs",
        "epoch_ica_spatial_pngs",
        "epoch_blob_summary_png",
        "epoch_blob_trace_pngs",
        "epoch_ica_converged",
        "epoch_ica_n_iter",
        "epoch_ica_max_iter",

        # XGBoost fallback result
        "fallback_xgb_model_mode",
        "fallback_xgb_predicted_multiclass_int",
        "fallback_xgb_predicted_multiclass_label",
        "fallback_xgb_predicted_binary",
        "fallback_xgb_predicted_binary_label",
        "fallback_xgb_p_dead",
        "fallback_xgb_p_alive",
        "fallback_xgb_p_weak",
        "fallback_xgb_p_strong",
        "fallback_xgb_alive_threshold",
    ]

    if len(plate_df) > 0 and "well" in plate_df.columns:
        plate_df["well"] = (
            plate_df["well"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        for col in mobility_columns:
            if col not in plate_df.columns:
                plate_df[col] = pd.NA

        plate_mobility_df = (
            plate_df[mobility_columns]
            .drop_duplicates(subset=["well"], keep="last")
            .copy()
        )

        summary_df = summary_df.merge(
            plate_mobility_df,
            on="well",
            how="outer",
        )

    else:
        for col in mobility_columns:
            if col != "well" and col not in summary_df.columns:
                summary_df[col] = pd.NA

    # ==========================================================
    # FINAL Alive/Dead classification
    # ==========================================================
    # plate_summary.csv is authoritative.
    #
    #   alive_by_motion            -> Alive / Mobility
    #   alive_by_slight_motion     -> Alive / Mobility
    #   alive_by_xgboost_fallback  -> Alive / XGBoost
    #   dead_by_xgboost_fallback   -> Dead  / XGBoost
    #   anything else              -> Unknown
    #
    # Heartbeat/blob outputs remain QC/descriptive only and do not
    # override the final motion/XGBoost classification.
    # ==========================================================

    summary_df["alive_dead"] = "Unknown"
    summary_df["assessed_by"] = "Unknown"

    if "mobility_alive_call" not in summary_df.columns:
        summary_df["mobility_alive_call"] = pd.NA

    summary_df["mobility_alive_call"] = (
        summary_df["mobility_alive_call"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    def flag_series(column_name):
        if column_name not in summary_df.columns:
            return pd.Series(False, index=summary_df.index)

        return (
            pd.to_numeric(
                summary_df[column_name],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )

    # Backward-compatible reconstruction only if mobility_alive_call
    # is absent/blank in an older plate_summary.csv.
    missing_call_mask = summary_df["mobility_alive_call"].eq("")

    summary_df.loc[
        missing_call_mask
        & flag_series("mobility_alive_by_strong_motion"),
        "mobility_alive_call",
    ] = "alive_by_motion"

    summary_df.loc[
        summary_df["mobility_alive_call"].eq("")
        & flag_series("mobility_alive_by_slight_motion"),
        "mobility_alive_call",
    ] = "alive_by_slight_motion"

    xgb_binary_label = (
        summary_df.get(
            "fallback_xgb_predicted_binary_label",
            pd.Series("", index=summary_df.index),
        )
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    summary_df.loc[
        summary_df["mobility_alive_call"].eq("")
        & xgb_binary_label.eq("alive"),
        "mobility_alive_call",
    ] = "alive_by_xgboost_fallback"

    summary_df.loc[
        summary_df["mobility_alive_call"].eq("")
        & xgb_binary_label.eq("dead"),
        "mobility_alive_call",
    ] = "dead_by_xgboost_fallback"

    call = summary_df["mobility_alive_call"]

    motion_alive_mask = call.isin([
        "alive_by_motion",
        "alive_by_slight_motion",
    ])
    xgb_alive_mask = call.eq("alive_by_xgboost_fallback")
    xgb_dead_mask = call.eq("dead_by_xgboost_fallback")

    summary_df.loc[motion_alive_mask, "alive_dead"] = "alive"
    summary_df.loc[motion_alive_mask, "assessed_by"] = "Mobility"

    summary_df.loc[xgb_alive_mask, "alive_dead"] = "alive"
    summary_df.loc[xgb_alive_mask, "assessed_by"] = "XGBoost"

    summary_df.loc[xgb_dead_mask, "alive_dead"] = "dead"
    summary_df.loc[xgb_dead_mask, "assessed_by"] = "XGBoost"

    # XGBoost subtype is authoritative for fallback wells.
    xgb_multi = (
        summary_df.get(
            "fallback_xgb_predicted_multiclass_label",
            pd.Series("", index=summary_df.index),
        )
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    # A final XGBoost Dead call must always have type_of_hb = dead.
    summary_df.loc[xgb_dead_mask, "type_of_hb"] = "dead"

    # For XGBoost Alive calls, use the collapsed model subtype.
    summary_df.loc[
        xgb_alive_mask & xgb_multi.eq("weak"),
        "type_of_hb",
    ] = "weak"

    summary_df.loc[
        xgb_alive_mask & xgb_multi.eq("strong"),
        "type_of_hb",
    ] = "strong"

    # Keep heartbeat-derived subtype only for non-XGBoost rows as QC/descriptive information.
    summary_df["type_of_hb"] = (
        summary_df["type_of_hb"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    def get_mobility_reason(row):
        tier = str(
            row.get("mobility_motion_tier", "")
        ).strip()

        if tier == "alive_by_motion":
            return "strong_motion"
        if tier == "alive_by_slight_motion":
            return "slight_motion"
        return "none"

    summary_df["mobility_alive_reason"] = summary_df.apply(
        get_mobility_reason,
        axis=1,
    )

    def get_fallback_status(row):
        reason = str(
            row.get("longest_epoch_fallback_reason", "")
        ).strip()

        if reason and reason.lower() not in {"nan", "<na>"}:
            return reason

        used = pd.to_numeric(
            row.get("used_longest_epoch_fallback", pd.NA),
            errors="coerce",
        )
        corrected = pd.to_numeric(
            row.get("fallback_motion_correction_used", pd.NA),
            errors="coerce",
        )

        if pd.notna(used) and int(used) == 1:
            if pd.notna(corrected) and int(corrected) == 1:
                return "completed_motion_corrected"
            return "completed_without_motion_correction"

        return "not_used"

    summary_df["fallback_status"] = summary_df.apply(
        get_fallback_status,
        axis=1,
    )

    def get_final_reason(row):
        call_value = str(
            row.get("mobility_alive_call", "")
        ).strip()

        if call_value:
            return call_value

        reason = str(
            row.get("longest_epoch_fallback_reason", "")
        ).strip()

        if reason and reason.lower() not in {"nan", "<na>"}:
            return reason

        return "Unknown"

    summary_df["final_classification_reason"] = summary_df.apply(
        get_final_reason,
        axis=1,
    )

    print(
        "Final classification counts:",
        summary_df["alive_dead"]
        .value_counts(dropna=False)
        .to_dict(),
    )

    # ==========================================================
    # Merge normalized-energy summaries into well-level summary
    # ==========================================================

    energy_files = sorted(
        root_path.rglob("*_normalized_energy.csv")
    )

    print(
        f"Found {len(energy_files)} normalized-energy CSV files."
    )

    energy_dfs = []

    for energy_file in energy_files:
        try:
            energy_df = pd.read_csv(energy_file)

            required_energy_cols = {
                "well",
                "ic",
                "original_normalized_energy",
            }

            if not required_energy_cols.issubset(
                energy_df.columns
            ):
                print(
                    f"Skipping normalized-energy CSV with "
                    f"missing columns: {energy_file}"
                )
                continue

            energy_df = energy_df[
                [
                    "well",
                    "ic",
                    "original_normalized_energy",
                ]
            ].copy()

            energy_df["well"] = (
                energy_df["well"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            energy_df["ic"] = pd.to_numeric(
                energy_df["ic"],
                errors="coerce",
            )

            energy_df["original_normalized_energy"] = (
                pd.to_numeric(
                    energy_df["original_normalized_energy"],
                    errors="coerce",
                )
            )

            energy_df = energy_df.dropna(
                subset=[
                    "well",
                    "ic",
                    "original_normalized_energy",
                ]
            )

            energy_df["ic"] = energy_df["ic"].astype(int)

            energy_dfs.append(energy_df)

        except Exception as e:
            print(
                f"Error reading normalized-energy CSV "
                f"{energy_file}: {e}"
            )

    if len(energy_dfs) > 0:

        merged_energy_df = pd.concat(
            energy_dfs,
            ignore_index=True,
        )

        # In case duplicate well/IC rows somehow exist,
        # keep the last encountered value.
        merged_energy_df = (
            merged_energy_df
            .drop_duplicates(
                subset=["well", "ic"],
                keep="last",
            )
        )

        # Keep IC0 through IC3
        merged_energy_df = merged_energy_df[
            merged_energy_df["ic"].between(0, 3)
        ].copy()

        # Pivot:
        #
        # well | ic0_normalized_energy | ic1_normalized_energy ...
        #
        energy_wide_df = (
            merged_energy_df
            .pivot(
                index="well",
                columns="ic",
                values="original_normalized_energy",
            )
            .rename(
                columns={
                    0: "ic0_normalized_energy",
                    1: "ic1_normalized_energy",
                    2: "ic2_normalized_energy",
                    3: "ic3_normalized_energy",
                }
            )
            .reset_index()
        )

        # Ensure all four columns exist even if an IC is missing.
        for col in [
            "ic0_normalized_energy",
            "ic1_normalized_energy",
            "ic2_normalized_energy",
            "ic3_normalized_energy",
        ]:
            if col not in energy_wide_df.columns:
                energy_wide_df[col] = pd.NA

        summary_df = summary_df.merge(
            energy_wide_df,
            on="well",
            how="left",
        )

        print(
            f"Merged normalized energy for "
            f"{energy_wide_df['well'].nunique()} wells."
        )

    else:
        print("No usable normalized-energy CSV files found.")

        for col in [
            "ic0_normalized_energy",
            "ic1_normalized_energy",
            "ic2_normalized_energy",
            "ic3_normalized_energy",
        ]:
            summary_df[col] = pd.NA


    # ----------------------------------------------------------
    # Final column order and sort
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # Final CLEAN well-level output
    # ----------------------------------------------------------
    # Keep only fields needed to interpret the final classification.
    # Detailed ICA/blob/mask/camera-motion QC remains available in the
    # original plate_summary.csv and per-well output folders.
    required_cols = [
        "well",

        # Final result
        "alive_dead",
        "type_of_hb",
        "assessed_by",
        "mobility_alive_call",
        "final_classification_reason",

        # Motion evidence
        "mobility_motion_tier",
        "n_strong_motion_peaks",
        "n_slight_motion_peaks",
        "max_pos_dev_from_baseline",

        # Fallback status
        "fallback_status",
        "longest_epoch_fallback_reason",

        # XGBoost result
        "fallback_xgb_model_mode",
        "fallback_xgb_predicted_multiclass_label",
        "fallback_xgb_predicted_binary_label",

        # XGBoost probabilities
        "fallback_xgb_p_dead",
        "fallback_xgb_p_alive",
        "fallback_xgb_p_weak",
        "fallback_xgb_p_strong",
    ]

    for c in required_cols:
        if c not in summary_df.columns:
            summary_df[c] = pd.NA

    summary_df = summary_df[required_cols].copy()

    summary_df = summary_df.sort_values(
        by="well",
        key=lambda s: s.map(well_sort_key),
    ).reset_index(drop=True)

    # ----------------------------------------------------------
    # Save well-level summary
    # ----------------------------------------------------------
    well_summary_csv = montage_dir / "merged_fallback_well_summary.csv"
    summary_df.to_csv(well_summary_csv, index=False)

    print(f"Final well count: {len(summary_df)}")
    print(f"Saved well-level summary CSV: {well_summary_csv}")
    print(f"Well summary shape: {summary_df.shape}")

    # ----------------------------------------------------------
    # Optional Excel export
    # ----------------------------------------------------------
    try:
        ic_summary_xlsx = montage_dir / "merged_fallback_ic_summary.xlsx"
        well_summary_xlsx = montage_dir / "merged_fallback_well_summary.xlsx"

        merged_df.to_excel(ic_summary_xlsx, index=False)
        summary_df.to_excel(well_summary_xlsx, index=False)

        print(f"Saved Excel IC/blob summary: {ic_summary_xlsx}")
        print(f"Saved Excel well summary: {well_summary_xlsx}")

    except Exception as e:
        print(f"Excel export skipped because it failed: {e}")
        print("CSV files were still saved successfully.")

    return merged_df, summary_df


# ==========================================================
# Main
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build heartbeat extractor montage PNGs and collate CSV summaries."
    )

    parser.add_argument(
        "root_path",
        type=str,
        help="Root output folder from heartbeat extractor.",
    )

    parser.add_argument(
        "--border-px",
        type=int,
        default=6,
        help="White border thickness in pixels.",
    )

    parser.add_argument(
        "--font-big",
        type=int,
        default=28,
        help="Outer label font size.",
    )

    parser.add_argument(
        "--font-small",
        type=int,
        default=36,
        help="Inner label font size.",
    )

    parser.add_argument(
        "--skip-montages",
        action="store_true",
        help="Only collate summaries; do not build montage PNGs.",
    )

    parser.add_argument(
        "--skip-collate",
        action="store_true",
        help="Only build montages; do not collate CSV summaries.",
    )

    args = parser.parse_args()

    root_path = Path(args.root_path)

    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")

    # ==========================================================
    # All outputs go here
    # ==========================================================
    montage_dir = root_path / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print("Heartbeat montage + collation builder")
    print(f"Root path   : {root_path}")
    print(f"Output dir  : {montage_dir}")
    print("==========================================================")

    # ==========================================================
    # Image types
    # ==========================================================
    image_types = {
        "spatial_maps": (
            "spatial_maps",
            "_epoch_spatialmaps_all.png",
        ),

        "blob_summary": (
            "blob_summary",
            "_epoch_blob_best.png",
        ),

        "inside_minus_outside_t2_epochs": (
            None,
            "_inside_minus_outside_t2_epochs.png",
        ),
    }

    # ==========================================================
    # Plate layout
    # ==========================================================
    rows = list(string.ascii_uppercase[:8])   # A-H
    cols = list(range(1, 13))                 # 1-12

    # ==========================================================
    # User knobs
    # ==========================================================
    border_px = args.border_px
    label_pad = 6

    border_color = (255, 255, 255)
    label_color = (0, 255, 0)

    inside_alpha_box = True

    font_big = load_font(args.font_big)
    font_small = load_font(args.font_small)

    # ==========================================================
    # Build montages
    # ==========================================================
    if not args.skip_montages:
        for image_type, (subdir, suffix) in image_types.items():
            build_montage(
                root_path=root_path,
                image_type=image_type,
                subdir=subdir,
                suffix=suffix,
                rows=rows,
                cols=cols,
                montage_dir=montage_dir,
                border_px=border_px,
                label_pad=label_pad,
                border_color=border_color,
                label_color=label_color,
                inside_alpha_box=inside_alpha_box,
                font_big=font_big,
                font_small=font_small,
            )
    else:
        print("Skipping montage generation.")

    # ==========================================================
    # Collate heartbeat CSV summaries
    # ==========================================================
    if not args.skip_collate:
        collate_heartbeat_data(
            root_path=root_path,
            montage_dir=montage_dir,
        )
    else:
        print("Skipping CSV collation.")

    print("\nDone.")
    print(f"All outputs saved in: {montage_dir}")


if __name__ == "__main__":
    main()