#!/usr/bin/env python

from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np


# ============================================================
# Feature file settings
# ============================================================

FEATURE_SPECS = {
    "morphology": {
        "folder_keywords": ["morphology"],
        "csv": "montages/lbend_all_mask_areas_merged.csv",
    },

    "tracking": {
        "folder_keywords": ["tracking"],
        "csv": "distance_traveled_key_metrics_processed.csv",
    },

    "hr": {
        "folder_keywords": ["hr", "hb"],
        "csv": "montages/merged_fallback_well_summary.csv",
    },

    "fluoro": {
        "folder_keywords": ["fluor"],
        "csv": "montages/summary_by_well.csv",
    },
}


# ============================================================
# Keep only these columns from each file
# "well" must stay in each list
# Source columns are included so you can debug which file was used
# ============================================================

KEEP_COLS = {
    "morphology": [
        "well",
        "source_path",
        "severity",
    ],

    "tracking": [
        "well",
        "source_path",
        "total_distance_traveled",
        "percent_missed",
        "peak_speed",
    ],

    "hr": [
        "well",
        "source_path",
        "alive_dead",
        "type_of_hb",
        "assessed_by",
        "n_peaks_real",
        "peak_signal_name",
        "is_periodic",
        "ic0_normalized_energy",
        "ic1_normalized_energy",
        "ic2_normalized_energy",
        "ic3_normalized_energy",
    ],

    "fluoro": [
        "well",
        "source_path",
        "top10_green_mean",
        "fluoro_top10_green_mean",
        "fluoro_top10_green_mean_adjusted",
        "plate_value_factor",
        "top10_green_std",
        "n_points",
        "bright_brain_by_top10_green",
        "classification_rule",
    ],
}


# ============================================================
# Helper functions
# ============================================================

def clean_colname(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
    )


def normalize_text(s):
    return (
        str(s)
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


# def folder_matches_feature(folder_name, feature_name):
#     s = normalize_text(folder_name)
#     keywords = FEATURE_SPECS[feature_name]["folder_keywords"]

#     return any(normalize_text(keyword) in s for keyword in keywords)
def folder_matches_feature(folder_name, feature_name):
    name = str(folder_name).lower()

    if feature_name == "morphology":
        return re.match(r"^\d+[a-z]*[_-]*morphology[_-]", name, flags=re.I) is not None

    if feature_name == "tracking":
        return re.match(r"^\d+[a-z]*[_-]*tracking[_-]", name, flags=re.I) is not None

    if feature_name == "hr":
        return re.match(r"^\d+[a-z]*[_-]*(hr|hb)[_-]", name, flags=re.I) is not None

    if feature_name == "fluoro":
        return re.match(r"^\d+[a-z]*[_-]*(fluor|fluoro)[_-]", name, flags=re.I) is not None

    return False


def strip_feature_prefix(folder_name):
    """
    Remove leading feature prefix before checking timepoint.

    This prevents false matches like:
        02_HR_Perampanel_30mins_...

    incorrectly matching 120 min because:
        120 min = 2hr
        02_HR can look like 2hr if normalized carelessly.
    """

    s = str(folder_name).strip()

    patterns = [
        r"^\d+[A-Z]*[_-]*morphology[_-]*",
        r"^\d+[A-Z]*[_-]*tracking[_-]*",
        r"^\d+[A-Z]*[_-]*HR[_-]*",
        r"^\d+[A-Z]*[_-]*HB[_-]*",
        r"^\d+[A-Z]*[_-]*fluoro[_-]*",
        r"^\d+[A-Z]*[_-]*fluor[_-]*",
    ]

    for pat in patterns:
        s2 = re.sub(pat, "", s, flags=re.I)
        if s2 != s:
            return s2

    s = re.sub(
        r"^\d+[A-Z]*[_-]*(?=(morphology|tracking|HR|HB|fluoro|fluor)[_-])",
        "",
        s,
        flags=re.I,
    )

    s = re.sub(
        r"^(morphology|tracking|HR|HB|fluoro|fluor)[_-]*",
        "",
        s,
        flags=re.I,
    )

    return s


def folder_matches_timepoint(folder_name, timepoint_min):
   
    tp = int(timepoint_min)

    cleaned = strip_feature_prefix(folder_name)

    # ------------------------------------------------------------
    # OLD LOGIC: token-based exact matching
    # ------------------------------------------------------------
    tokens = re.split(r"[_\-\s]+", cleaned)
    tokens = [t.strip().lower() for t in tokens if t.strip() != ""]

    minute_patterns = {
        f"{tp}min",
        f"{tp}mins",
        f"{tp}minute",
        f"{tp}minutes",
    }

    if any(t in minute_patterns for t in tokens):
        return True

    if tp % 60 == 0:
        hr = tp // 60

        hour_patterns = {
            f"{hr}hr",
            f"{hr}hrs",
            f"{hr}hour",
            f"{hr}hours",
        }

        if any(t in hour_patterns for t in tokens):
            return True

    # ------------------------------------------------------------
    # NEW LOGIC: allow spaces between number and unit
    # Example: "120 min", "2 hr"
    # ------------------------------------------------------------
    s = str(cleaned).lower()

    min_pattern = rf"(?<!\d){tp}\s*(min|mins|minute|minutes)(?![a-z])"

    if re.search(min_pattern, s, flags=re.I):
        return True

    if tp % 60 == 0:
        hr = tp // 60

        hr_pattern = rf"(?<!\d){hr}\s*(hr|hrs|hour|hours)(?![a-z])"

        if re.search(hr_pattern, s, flags=re.I):
            return True

    return False


def get_source_drug_folder(source_folder):
    s = str(source_folder)

    s = re.sub(r"^(01B_)?tracking_", "", s, flags=re.I)
    s = re.sub(r"^(01_)?morphology_", "", s, flags=re.I)
    s = re.sub(r"^(02_)?(HR|HB)_", "", s, flags=re.I)
    s = re.sub(r"^(03_)?fluoro_", "", s, flags=re.I)
    s = re.sub(r"^(03_)?fluor_", "", s, flags=re.I)

    parts = s.split("_")

    drug_parts = []
    timepoint = None
    date = None

    for part in parts:
        part_clean = part.strip()

        if part_clean == "":
            continue

        m_min = re.match(r"^(\d+)\s*(min|mins|minute|minutes)$", part_clean, flags=re.I)
        if m_min:
            timepoint = f"{int(m_min.group(1))}min"
            continue

        m_hr = re.match(r"^(\d+)\s*(hr|hrs|hour|hours)$", part_clean, flags=re.I)
        if m_hr:
            minutes = int(m_hr.group(1)) * 60
            timepoint = f"{minutes}min"
            continue

        if re.match(r"^20\d{6}$", part_clean):
            date = part_clean
            break

        drug_parts.append(part_clean)

    if len(drug_parts) > 0 and timepoint is not None and date is not None:
        drug = "_".join(drug_parts)
        return f"{drug}_{timepoint}_{date}"

    return source_folder


def case_insensitive_existing_path(path):
    path = Path(path)

    if path.exists():
        return path

    parts = path.parts

    if path.is_absolute():
        current = Path(parts[0])
        remaining_parts = parts[1:]
    else:
        current = Path(".")
        remaining_parts = parts

    for part in remaining_parts:
        if not current.exists() or not current.is_dir():
            return path

        matches = [
            p for p in current.iterdir()
            if p.name.lower() == part.lower()
        ]

        if len(matches) == 0:
            return path

        current = matches[0]

    return current


def find_feature_folder(root_dir, feature_name, timepoint_min):
    root_dir = Path(root_dir)

    candidates = []

    for p in root_dir.iterdir():
        if not p.is_dir():
            continue

        if (
            folder_matches_feature(p.name, feature_name)
            and folder_matches_timepoint(p.name, timepoint_min)
        ):
            candidates.append(p)

    candidates = sorted(candidates)

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print()
        print("=" * 100)
        print(f"WARNING: multiple folders found for {feature_name}, {timepoint_min} min.")
        print("These folders all match the same feature and timepoint:")
        for c in candidates:
            print("  ", c)
        print()
        print("Using first one after sorting:")
        print("  ", candidates[0])
        print("=" * 100)
        print()

    return candidates[0]


def find_csv_for_feature(root_dir, feature_name, timepoint_min):
    folder = find_feature_folder(root_dir, feature_name, timepoint_min)

    if folder is None:
        print(f"WARNING: no folder found for {feature_name}, {timepoint_min} min")
        return None

    csv_path = folder / FEATURE_SPECS[feature_name]["csv"]
    csv_path = case_insensitive_existing_path(csv_path)

    if not csv_path.exists():
        print(f"WARNING: CSV not found for {feature_name}, {timepoint_min} min:")
        print(" ", csv_path)
        return None

    return csv_path


def standardize_well_name(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    m = re.search(r"(?:well_)?([A-Ha-h]\d{1,2})", s, flags=re.I)

    if m:
        well = m.group(1).upper()
        row = well[0]
        col = int(well[1:])
        return f"{row}{col}"

    return s


def find_well_column(df):
    possible = ["well", "wells", "well_id", "well_name"]

    for c in df.columns:
        if str(c).strip().lower() in possible:
            return c

    if "base_name" in df.columns:
        return "base_name"

    first_col = df.columns[0]
    vals = df[first_col].dropna().astype(str).head(30)

    if vals.str.match(r"^(well_)?[A-Ha-h]\d{1,2}$", case=False).any():
        print(f"Using first column as well column: {first_col}")
        return first_col

    raise ValueError(f"Could not find well column. Columns are: {list(df.columns)}")


def add_or_fix_well_column(df):
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    if "well" in df.columns:
        df["well"] = df["well"].apply(standardize_well_name)
        return df

    well_col = find_well_column(df)
    df["well"] = df[well_col].apply(standardize_well_name)

    return df


def keep_selected_columns(df, keep_cols, label=""):
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    keep_cols_clean = [clean_colname(c) for c in keep_cols]

    existing = [c for c in keep_cols_clean if c in df.columns]
    missing = [c for c in keep_cols_clean if c not in df.columns]

    if missing:
        print(f"WARNING: missing columns in {label}:")
        print(missing)
        print("Available columns:")
        print(list(df.columns))

    if "well" not in existing:
        existing = ["well"] + existing

    existing = list(dict.fromkeys(existing))

    return df[existing]


def get_source_folder_from_csv(csv_path):
    csv_path = Path(csv_path)

    if csv_path.parent.name.lower() == "montages":
        return csv_path.parent.parent.name

    return csv_path.parent.name


def read_feature_csv(csv_path, feature_name, timepoint_min):
    csv_path = Path(csv_path)

    print(f"\nReading {feature_name}_{timepoint_min}")
    print(" ", csv_path)

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [clean_colname(c) for c in df.columns]

    df = add_or_fix_well_column(df)
    df = df[df["well"].notna()].copy()

    source_folder = get_source_folder_from_csv(csv_path)
    source_drug_folder = get_source_drug_folder(source_folder)

    df.insert(0, "timepoint_min", timepoint_min)
    df.insert(1, "feature_type", feature_name)
    df.insert(2, "source_drug_folder", source_drug_folder)
    df.insert(3, "source_folder", source_folder)
    df.insert(4, "source_file", csv_path.name)
    df.insert(5, "source_path", str(csv_path))

    return df


def sort_by_well_plate_order(df):
    df = df.copy()

    well_values = df["well"].astype(str).str.upper()

    row_letter = well_values.str.extract(r"([A-H])", expand=False)
    col_number = well_values.str.extract(r"([0-9]+)", expand=False)

    df["_row_sort"] = row_letter.values
    df["_col_sort"] = pd.to_numeric(col_number.values, errors="coerce")

    df = (
        df.sort_values(["_row_sort", "_col_sort"])
        .drop(columns=["_row_sort", "_col_sort"])
        .reset_index(drop=True)
    )

    return df


def check_duplicate_wells(df, feature_name, timepoint_min):
    if "well" not in df.columns:
        return

    dup_mask = df["well"].duplicated(keep=False)

    if dup_mask.any():
        dup_wells = sorted(df.loc[dup_mask, "well"].dropna().astype(str).unique())

        print()
        print("=" * 100)
        print(f"WARNING: duplicate wells found in {feature_name}_{timepoint_min}")
        print("Duplicate wells:")
        print(dup_wells)
        print("This can cause duplicated rows after merge.")
        print("=" * 100)
        print()


def prefix_non_well_columns(df, feature_name):
    """
    Prefix non-well columns with the feature name, but do not add the
    prefix again when the source column already starts with it.

    Example:
        top10_green_mean -> fluoro_top10_green_mean
        fluoro_top10_green_mean_adjusted -> unchanged
    """
    prefix = f"{feature_name}_"

    rename_map = {
        c: c if c.startswith(prefix) else f"{prefix}{c}"
        for c in df.columns
        if c != "well"
    }

    return df.rename(columns=rename_map)


def join_one_timepoint(root_dir, timepoint_min):
    """
    For one timepoint:
    - find morphology/tracking/hr/fluoro CSVs
    - keep only selected columns plus source-debug columns
    - join by well
    - save one final kept-column file
    """

    feature_order = ["morphology", "tracking", "hr", "fluoro"]

    joined = None
    source_drug_values = []

    for feature_name in feature_order:
        csv_path = find_csv_for_feature(root_dir, feature_name, timepoint_min)

        if csv_path is None:
            print(f"Skipping {feature_name}_{timepoint_min}")
            continue

        df = read_feature_csv(
            csv_path=csv_path,
            feature_name=feature_name,
            timepoint_min=timepoint_min,
        )

        check_duplicate_wells(df, feature_name, timepoint_min)

        vals = df["source_drug_folder"].dropna().astype(str).unique()

        if len(vals) > 0:
            source_drug_values.append(vals[0])

        df_small = keep_selected_columns(
            df,
            KEEP_COLS.get(feature_name, ["well"]),
            label=f"{feature_name}_{timepoint_min}",
        )

        df_small = prefix_non_well_columns(df_small, feature_name)

        if joined is None:
            joined = df_small
        else:
            joined = joined.merge(df_small, on="well", how="outer")

    if joined is None:
        print(f"No usable dataframes for {timepoint_min} min")
        return None, None

    if len(source_drug_values) > 0:
        source_drug_folder = source_drug_values[0]
    else:
        source_drug_folder = f"Final_{timepoint_min}mins_results"

    joined.insert(0, "source_drug_folder", source_drug_folder)
    joined = sort_by_well_plate_order(joined)

    return joined, source_drug_folder


def process_all_timepoints(root_dir, timepoints, output_dir=None):
    root_dir = Path(root_dir)

    if output_dir is None:
        out_dir = root_dir / "Final_results"
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Final timepoint join processor")
    print("=" * 100)
    print("ROOT_DIR:", root_dir)
    print("OUT_DIR  :", out_dir)
    print("TIMEPOINTS:", timepoints)

    for tp in timepoints:
        print("\n" + "#" * 100)
        print(f"Processing {tp} min")
        print("#" * 100)

        joined, source_drug_folder = join_one_timepoint(
            root_dir=root_dir,
            timepoint_min=tp,
        )

        if joined is None:
            print(f"Skipping save for {tp} min because no joined dataframe was created.")
            continue

        out_path = out_dir / f"{source_drug_folder}_KEPT_COLUMNS.csv"
        joined.to_csv(out_path, index=False)

        print()
        print("Saved:")
        print(f"  {out_path}")
        print(f"Shape: {joined.shape}")

        print()
        print("Source paths used for this timepoint:")

        source_cols = [
            c for c in joined.columns
            if c.endswith("_source_path")
        ]

        for c in source_cols:
            vals = joined[c].dropna().astype(str).unique()
            if len(vals) > 0:
                print(f"  {c}:")
                print(f"    {vals[0]}")

    print("\nDone.")


# ============================================================
# Command line
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Join morphology, tracking, HR/HB, and fluor CSVs for selected "
            "timepoints and save one kept-column output per timepoint."
        )
    )

    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root folder containing morphology, tracking, HR/HB, and fluor folders.",
    )

    parser.add_argument(
        "--output_dir",
        required=False,
        default=None,
        help="Folder where final joined CSV files will be saved. Default: ROOT_DIR/Final_results",
    )

    parser.add_argument(
        "--timepoints",
        nargs="+",
        type=int,
        default=[30, 60, 120, 180],
        help="Timepoints to process. Default: 30 60 120 180",
    )

    args = parser.parse_args()

    process_all_timepoints(
        root_dir=args.root_dir,
        timepoints=args.timepoints,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()