#!/usr/bin/env python

from pathlib import Path
import argparse
import string

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    """
    Try common fonts on Linux/Quest. Fall back to default if unavailable.
    """
    font_paths = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    ]

    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass

    return ImageFont.load_default()


def make_montage_for_folder(
    folder_path: Path,
    output_path: Path,
    image_suffix: str,
    *,
    thumb_w: int = 500,
    thumb_h: int = 500,
    border_px: int = 4,
    label_color=(0, 255, 0),
    border_color=(255, 255, 255),
    background_color=(0, 0, 0),
):
    rows = list(string.ascii_uppercase[:8])   # A-H
    cols = list(range(1, 13))                 # 1-12

    label_pad = 6
    tile_w = thumb_w + 2 * border_px
    tile_h = thumb_h + 2 * border_px

    font_outer = load_font(28)
    font_inner = load_font(36)

    print("\n------------------------------------------")
    print(f"Input folder : {folder_path}")
    print(f"Image suffix : {image_suffix}")
    print(f"Output path  : {output_path}")
    print("------------------------------------------")

    if not folder_path.exists():
        print(f"Folder does not exist, skipping: {folder_path}")
        return {
            "folder": str(folder_path),
            "output": str(output_path),
            "found": 0,
            "missing": 96,
            "status": "folder_missing",
        }

    montage = Image.new(
        "RGB",
        (tile_w * 12, tile_h * 8),
        background_color,
    )

    missing_count = 0
    found_count = 0
    error_count = 0

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            well = f"{row}{col}"
            filename = f"well_{well}{image_suffix}"
            filepath = folder_path / filename

            x0 = j * tile_w
            y0 = i * tile_h

            if not filepath.exists():
                print(f"Missing: {filename}")
                missing_count += 1

                # Empty placeholder tile
                tile = Image.new("RGB", (tile_w, tile_h), border_color)
                img_canvas = Image.new("RGB", (thumb_w, thumb_h), background_color)
                tile.paste(img_canvas, (border_px, border_px))

                draw = ImageDraw.Draw(tile)
                draw.text(
                    (label_pad, label_pad),
                    f"{well}\nMISSING",
                    font=font_outer,
                    fill=(255, 0, 0),
                )

                montage.paste(tile, (x0, y0))
                continue

            try:
                img = Image.open(filepath).convert("RGB")
            except Exception as e:
                print(f"Could not open {filepath}: {e}")
                error_count += 1

                tile = Image.new("RGB", (tile_w, tile_h), border_color)
                img_canvas = Image.new("RGB", (thumb_w, thumb_h), background_color)
                tile.paste(img_canvas, (border_px, border_px))

                draw = ImageDraw.Draw(tile)
                draw.text(
                    (label_pad, label_pad),
                    f"{well}\nERROR",
                    font=font_outer,
                    fill=(255, 0, 0),
                )

                montage.paste(tile, (x0, y0))
                continue

            found_count += 1

            # Resize without cropping
            img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)

            img_canvas = Image.new("RGB", (thumb_w, thumb_h), background_color)
            px = (thumb_w - img.width) // 2
            py = (thumb_h - img.height) // 2
            img_canvas.paste(img, (px, py))

            tile = Image.new("RGB", (tile_w, tile_h), border_color)
            tile.paste(img_canvas, (border_px, border_px))

            draw = ImageDraw.Draw(tile)

            # Outer label: top-left
            outer_xy = (label_pad, label_pad)

            # Black outline
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text(
                    (outer_xy[0] + dx, outer_xy[1] + dy),
                    well,
                    font=font_outer,
                    fill=(0, 0, 0),
                )

            draw.text(
                outer_xy,
                well,
                font=font_outer,
                fill=label_color,
            )

            # Inner label: bottom-left with black box
            inner_x = border_px + 10
            inner_y = border_px + thumb_h - 50

            draw.rectangle(
                [inner_x - 5, inner_y - 5, inner_x + 105, inner_y + 45],
                fill=(0, 0, 0),
            )

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text(
                    (inner_x + dx, inner_y + dy),
                    well,
                    font=font_inner,
                    fill=(0, 0, 0),
                )

            draw.text(
                (inner_x, inner_y),
                well,
                font=font_inner,
                fill=label_color,
            )

            montage.paste(tile, (x0, y0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path)

    print(f"Saved montage: {output_path}")
    print(f"Found images : {found_count}")
    print(f"Missing      : {missing_count}")
    print(f"Errors       : {error_count}")

    return {
        "folder": str(folder_path),
        "output": str(output_path),
        "found": found_count,
        "missing": missing_count,
        "errors": error_count,
        "status": "done",
    }


def merge_csvs(root_path: Path, output_folder: Path):
    """
    Merge all CSV files ending with _mask_areas.csv from:

        root_path / "csv"

    and save the merged CSV inside:

        root_path / "montages"
    """

    csv_dir = root_path / "csv"

    print("\n------------------------------------------")
    print("Merging L-bend CSV files")
    print(f"CSV folder    : {csv_dir}")
    print(f"Output folder : {output_folder}")
    print("------------------------------------------")

    if not csv_dir.exists():
        print(f"CSV folder does not exist, skipping summary: {csv_dir}")
        return None

    # Finds:
    # well_A1_mask_areas.csv
    # well_B12_mask_areas.csv
    # anything_else_mask_areas.csv
    csv_files = sorted(csv_dir.glob("*_mask_areas.csv"))

    print(f"CSV files found: {len(csv_files)}")

    if len(csv_files) == 0:
        print("No *_mask_areas.csv files found.")
        return None

    dfs = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Add source file for traceability
            df["source_csv"] = csv_file.name

            dfs.append(df)

            print(f"Read {csv_file.name}: {len(df)} rows")

        except Exception as e:
            print(f"Could not read {csv_file}: {e}")

    if len(dfs) == 0:
        print("No CSV files could be read.")
        return None

    merged = pd.concat(dfs, ignore_index=True)

    output_folder.mkdir(parents=True, exist_ok=True)

    merged_csv = output_folder / "lbend_all_mask_areas_merged.csv"
    merged.to_csv(merged_csv, index=False)

    print("------------------------------------------")
    print(f"✅ Saved merged CSV: {merged_csv}")
    print(f"Total rows: {len(merged)}")
    print("------------------------------------------")

    return merged_csv


def main():
    parser = argparse.ArgumentParser(
        description="Make L-bend 96-well montages and merge per-well CSVs."
    )

    parser.add_argument(
        "--root_path",
        required=True,
        help="Plate output root. Example: /path/to/processed/<plate_name>",
    )

    parser.add_argument("--thumb_w", type=int, default=500)
    parser.add_argument("--thumb_h", type=int, default=500)
    parser.add_argument("--border_px", type=int, default=4)

    args = parser.parse_args()

    root_path = Path(args.root_path)
    output_folder = root_path / "montages"
    output_folder.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print("L-bend montage + CSV merge")
    print(f"Root path     : {root_path}")
    print(f"Output folder : {output_folder}")
    print("==========================================================")

    image_types = {
        "overlays": ("overlays", "_f0000_overlay.png"),
        "frames": ("png", "_f0000_frame.png"),
        "skeleton": ("figures_skeleton", "_f0000_skeleton_on_binary.png"),
        "polyfit": ("figures_fit", "_f0000_polyfit_debug.png"),
        "poly_midline": ("figures_midline", "_f0000_poly_midline.png"),
    }

    montage_records = []

    for name, (subfolder, suffix) in image_types.items():
        folder_path = root_path / subfolder
        output_path = output_folder / f"montage_f0000_{name}.png"

        rec = make_montage_for_folder(
            folder_path=folder_path,
            output_path=output_path,
            image_suffix=suffix,
            thumb_w=args.thumb_w,
            thumb_h=args.thumb_h,
            border_px=args.border_px,
        )

        rec["image_type"] = name
        rec["subfolder"] = subfolder
        rec["suffix"] = suffix
        montage_records.append(rec)

    montage_log = pd.DataFrame(montage_records)
    montage_log_path = output_folder / "montage_generation_log.csv"
    montage_log.to_csv(montage_log_path, index=False)

    print("------------------------------------------")
    print(f"✅ Saved montage generation log: {montage_log_path}")
    print("------------------------------------------")

    merge_csvs(root_path, output_folder)

    print("\n✅ Done making L-bend montages.")
    print(f"Montages saved in: {output_folder}")
    print(f"Montage log saved: {montage_log_path}")


if __name__ == "__main__":
    main()