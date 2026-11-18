#!/usr/bin/env python

from pathlib import Path
import argparse
import csv
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def clean_text(x):
    s = str(x).strip()

    if s == "" or s.lower() in ["nan", "none"] or s.lower().startswith("unnamed:"):
        return ""

    return (
        s.replace(" ", "_")
         .replace("-", "_")
         .replace("/", "_")
         .replace("(", "")
         .replace(")", "")
         .replace("[", "")
         .replace("]", "")
         .replace(":", "")
         .strip("_")
    )


def sort_wells(df):
    if "well" not in df.columns:
        return df

    tmp = df.copy()

    tmp["row_sort"] = tmp["well"].astype(str).str.extract(r"well_([A-H])")[0]
    tmp["col_sort"] = tmp["well"].astype(str).str.extract(r"well_[A-H](\d+)")[0]
    tmp["col_sort"] = pd.to_numeric(tmp["col_sort"], errors="coerce")

    tmp = (
        tmp.sort_values(["row_sort", "col_sort"])
           .drop(columns=["row_sort", "col_sort"])
           .reset_index(drop=True)
    )

    return tmp


def read_ragged_csv(input_csv):
    """
    Read CSV files where different rows have different numbers of columns.
    This avoids pandas tokenizing errors like:
    Expected 2 fields in line 10, saw 14
    """

    input_csv = Path(input_csv)

    rows = []

    with open(input_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        raise ValueError(f"Empty file: {input_csv}")

    max_cols = max(len(row) for row in rows)

    padded_rows = []
    for row in rows:
        padded = row + [""] * (max_cols - len(row))
        padded_rows.append(padded)

    return pd.DataFrame(padded_rows)


# ============================================================
# Processor:
# distance_traveled_key_metrics.csv only
# ============================================================

def process_distance_traveled_key_metrics(input_csv, output_csv):
    """
    Process distance_traveled_key_metrics.csv.

    Does NOT use skiprows=11.

    It automatically finds the row where first column is 'row',
    then uses:
        row
        column
        total_distance_traveled
        percent_missed
        peak_speed

    Output:
        well,total_distance_traveled,percent_missed,peak_speed
    """

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    print(f"\nProcessing key metrics:")
    print(f"  Input : {input_csv}")
    print(f"  Output: {output_csv}")

    # ------------------------------------------------------------
    # Read ragged CSV safely
    # ------------------------------------------------------------
    raw = read_ragged_csv(input_csv)

    # ------------------------------------------------------------
    # Find table start: first column equals "row"
    # ------------------------------------------------------------
    first_col = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    row_matches = raw.index[first_col == "row"].tolist()

    if len(row_matches) == 0:
        raise ValueError(f"Could not find table row starting with 'row' in: {input_csv}")

    table_start = row_matches[0]

    table = raw.iloc[table_start:].reset_index(drop=True)

    labels = table.iloc[:, 0].astype(str).str.strip()
    labels_lower = labels.str.lower()

    # ------------------------------------------------------------
    # Locate row and column rows
    # ------------------------------------------------------------
    row_row_matches = table.index[labels_lower == "row"].tolist()
    col_row_matches = table.index[labels_lower == "column"].tolist()

    if len(row_row_matches) == 0:
        raise ValueError(f"Could not find 'row' row in: {input_csv}")

    if len(col_row_matches) == 0:
        raise ValueError(f"Could not find 'column' row in: {input_csv}")

    row_row = row_row_matches[0]
    col_row = col_row_matches[0]

    # ------------------------------------------------------------
    # Locate metric rows
    # ------------------------------------------------------------
    metric_names_to_keep = [
        "total_distance_traveled",
        "percent_missed",
        "peak_speed",
    ]

    metric_rows = []

    for idx, label in enumerate(labels):
        cleaned = clean_text(label).lower()
        if cleaned in metric_names_to_keep:
            metric_rows.append(idx)

    if len(metric_rows) == 0:
        raise ValueError(f"No metric rows found in: {input_csv}")

    # ------------------------------------------------------------
    # Build well names from row/column rows
    # ------------------------------------------------------------
    well_rows = table.iloc[row_row, 1:]
    well_cols = table.iloc[col_row, 1:]

    wells = []
    valid_positions = []

    for pos, (r, c) in enumerate(zip(well_rows, well_cols), start=1):
        r = str(r).strip()
        c = str(c).strip()

        if r == "" or c == "":
            continue

        if r.lower() == "nan" or c.lower() == "nan":
            continue

        try:
            c_num = int(float(c))
        except Exception:
            continue

        wells.append(f"well_{r}{c_num}")
        valid_positions.append(pos)

    if len(wells) == 0:
        raise ValueError(f"No wells found from row/column rows in: {input_csv}")

    # ------------------------------------------------------------
    # Build output dataframe
    # ------------------------------------------------------------
    out = pd.DataFrame({"well": wells})

    for mr in metric_rows:
        metric_name = clean_text(table.iloc[mr, 0])

        values = []
        for pos in valid_positions:
            values.append(table.iloc[mr, pos])

        out[metric_name] = pd.to_numeric(values, errors="coerce")

    out = sort_wells(out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"  Saved: {output_csv}")
    print(f"  Shape: {out.shape}")

    return out


# ============================================================
# Folder processing
# ============================================================

def find_tracking_folders(root_dir):
    """
    Find folders whose folder name contains 'tracking'.
    Only returns folders that contain a results/ subfolder.
    """

    root_dir = Path(root_dir)

    tracking_folders = []

    for p in root_dir.iterdir():
        if not p.is_dir():
            continue

        if "tracking" not in p.name.lower():
            continue

        results_dir = p / "results"

        if not results_dir.exists():
            print(f"Skipping {p.name}: no results folder")
            continue

        tracking_folders.append(p)

    return sorted(tracking_folders)


def process_one_tracking_folder(tracking_folder, output_root):
    """
    Process one tracking folder.

    Input:
        ROOT_DIR/some_tracking_folder/results/distance_traveled_key_metrics.csv

    Output:
        OUTPUT_ROOT/some_tracking_folder/
            distance_traveled_key_metrics_processed.csv
    """

    tracking_folder = Path(tracking_folder)
    output_root = Path(output_root)

    results_dir = tracking_folder / "results"

    out_dir = output_root / tracking_folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"Tracking folder: {tracking_folder}")
    print(f"Results folder : {results_dir}")
    print(f"Output folder  : {out_dir}")
    print("=" * 100)

    key_csv = results_dir / "distance_traveled_key_metrics.csv"

    processed = 0
    failed = 0

    if key_csv.exists():
        try:
            process_distance_traveled_key_metrics(
                input_csv=key_csv,
                output_csv=out_dir / "distance_traveled_key_metrics_processed.csv"
            )
            processed += 1
        except Exception as e:
            failed += 1
            print(f"FAILED key metrics for {tracking_folder.name}: {e}")
    else:
        print(f"Missing: {key_csv}")

    print(f"\nFinished {tracking_folder.name}")
    print(f"  Processed: {processed}")
    print(f"  Failed   : {failed}")


def process_all_tracking_folders(root_dir, output_dir):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Searching for tracking folders")
    print("=" * 100)
    print(f"ROOT_DIR  : {root_dir}")
    print(f"OUTPUT_DIR: {output_dir}")

    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root_dir}")

    tracking_folders = find_tracking_folders(root_dir)

    print(f"\nFound {len(tracking_folders)} tracking folders:")

    for f in tracking_folders:
        print(f"  - {f.name}")

    if len(tracking_folders) == 0:
        print("\nNo tracking folders found.")
        return

    for tracking_folder in tracking_folders:
        process_one_tracking_folder(
            tracking_folder=tracking_folder,
            output_root=output_dir
        )

    print("\n" + "=" * 100)
    print("ALL DONE")
    print(f"Output saved inside: {output_dir}")
    print("=" * 100)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find subfolders with 'tracking' in the folder name, then process only "
            "distance_traveled_key_metrics.csv inside each folder's results/ directory. "
            "This version handles ragged metadata rows and does not use skiprows."
        )
    )

    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root folder containing the tracking subfolders."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output root folder. A subfolder will be created for each tracking folder."
    )

    args = parser.parse_args()

    process_all_tracking_folders(
        root_dir=args.root_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()