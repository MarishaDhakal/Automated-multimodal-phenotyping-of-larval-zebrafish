# Automated Multimodal Phenotyping of Larval Zebrafish

Code accompanying the paper *"Automated multimodal phenotyping of larval
zebrafish."* This repository contains the production image-analysis pipelines that
extract **cardiac (heartbeat) viability**, **spinal / body-curvature (L-bend)
severity**, and **neural activity (GCaMP fluorescence)** phenotypes from
multi-camera array microscope (MCAM) recordings, together with a shared
Segment Anything (SAM)-based well/arena localization and object-selection layer.

This release is based on the analysis scripts used for the paper, with
configuration changes for running outside the original cluster.

---

## Overview

Three independent extractors share a common SAM-based well-localization and
mask-selection core (`sam_pipeline/sam_utils.py`):

| Pipeline | Entry script | Phenotype | Montage builder |
|----------|--------------|-----------|-----------------|
| **Heartbeat** | `sam_pipeline/heartbeat_pipeline_xgboost_CLEAN.py` | Cardiac viability (Alive/Dead; Weak/Strong) via motion + ICA periodicity features and an XGBoost classifier | `make_heartbeat_montages_FINAL_CLEAN.py` |
| **L-bend** | `sam_pipeline/lbend_seveirity.py` | Body-axis curvature severity from a skeletonized midline | `make_lbend_montages.py` |
| **GCaMP** | `sam_pipeline/g_camp.py` | Fluorescence intensity (top-10% green, plate-corrected) and bright-brain classification | `montage_adjusted_bright_brain.py` |

A plate-level **tracking summary** (`tracking_all_tp.py`) and a **final
aggregation** step (`final_join_with_adjusted_fluoro.py`) join the per-well
outputs into combined tables.

---

## Repository structure

```
.
├── README.md
├── requirements.txt            # pip dependencies (pinned to production versions)
├── environment.yml             # conda environment specification
├── sam_pipeline/               # all Python modules (import package)
│   ├── sam_utils.py            # shared SAM config + well/arena localization + mask selectors
│   ├── fluro_utils.py          # fluorescence localization helpers (GCaMP)
│   ├── heartbeat_utils.py      # heartbeat I/O + helper utilities
│   ├── skeletonize_utils_old.py# midline skeletonization + severity (L-bend)
│   ├── heartbeat_pipeline_xgboost_CLEAN.py
│   ├── make_heartbeat_montages_FINAL_CLEAN.py
│   ├── lbend_seveirity.py
│   ├── make_lbend_montages.py
│   ├── g_camp.py
│   ├── montage_adjusted_bright_brain.py
│   ├── tracking_all_tp.py
│   └── final_join_with_adjusted_fluoro.py
├── slurm_files/                # SLURM batch scripts + top-level launcher
│   ├── submit_all_new_data.sh  # launcher: submits all pipelines for a plate set
│   ├── mobility_hb.sbatch      # heartbeat extractor (array 1–96 wells)
│   ├── run_heartbeat_postprocessing.sbatch
│   ├── l_bend.sbatch           # L-bend extractor (array 1–96 wells)
│   ├── run_lbend_postprocessing.sbatch
│   ├── g_camp.sbatch           # GCaMP extractor (array 1–96 wells)
│   ├── run_gcamp_montage.sbatch
│   ├── run_tracking_summary.sbatch
│   └── final_result.sbatch
├── models/                     # trained classifier artifacts (see models/README.md)
│   ├── alive_weak_strong_xgb.json
│   ├── alive_dead_binary_xgb.json
│   └── alive_weak_strong_xgb_features.txt
└── checkpoints/                # SAM checkpoints NOT included — see checkpoints/README.md
    └── README.md
```

---

## Installation

Python **3.10** is required. A CPU-only PyTorch build is used in production.

### Option A — conda

```bash
conda env create -f environment.yml
conda activate mcam_phenotyping
```

### Option B — pip / venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# torch is pulled from the PyTorch CPU index:
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

### SAM checkpoint

Download the ViT-H SAM checkpoint into `checkpoints/` — see
[`checkpoints/README.md`](checkpoints/README.md).

---

## Running the pipeline

### Full launcher (SLURM)

The launcher discovers plate folders under a single input root, submits all four
extractors as SLURM array jobs (one task per 96-well plate well), their montage
post-processing jobs, and a final aggregation job that depends on all of them:

```bash
bash slurm_files/submit_all_new_data.sh --root /path/to/plate_root
```

Input plate folders are matched by name:

| Pipeline | Folder name pattern |
|----------|---------------------|
| Heartbeat | `*_HR_*` or `*_HB_*` |
| L-bend | `*morphology*` |
| GCaMP | `*fluor*` |

Extractor outputs are written under `<plate_root>/gittest/`, with a reusable
per-well cache (ring/arena parameters) under `<plate_root>/reusable_cache/`.
Final tables go to `<plate_root>/final_results/` by default; set
`FINAL_OUTPUT_DIR` to an absolute path to override it.

### Single extractor (example)

Each extractor is a standalone script. For example, the heartbeat extractor for a
single well:

```bash
python sam_pipeline/heartbeat_pipeline_xgboost_CLEAN.py \
    --plate_dir  /path/to/<plate>_HR_<...> \
    --out_root   /path/to/output \
    --well       A1 \
    --crop_size  1048 \
    --lower_area 8000 --upper_area 35000 \
    --checkpoint /path/to/checkpoints/sam_vit_h_4b8939.pth
```

Run any script with `--help` for the full argument list.

---

## Cluster configuration

Activate the Python environment before running the launcher. SLURM jobs inherit
it, so the environment and checkout must be accessible from the compute nodes.
The batch files use your site's default account and partition unless you set
SLURM's `SBATCH_ACCOUNT` and `SBATCH_PARTITION` environment variables:

```bash
conda activate mcam_phenotyping
SBATCH_ACCOUNT=your_account SBATCH_PARTITION=your_partition \
    bash slurm_files/submit_all_new_data.sh --root /path/to/plate_root
```

The launcher passes absolute paths to the Python scripts. If you submit a batch
file directly, provide the variables it requires (including `PY_SCRIPT` or
`MONTAGE_SCRIPT`) and create its `logs/` or `logs4/` directory before submission.

The heartbeat model is loaded from [`models/`](models/). Download the SAM
checkpoint as described in [`checkpoints/README.md`](checkpoints/README.md), or
set `SAM_CHECKPOINT` to its absolute path on a filesystem visible to compute
nodes.

---

## Key parameters (as run)

Shared **SAM** mask generator (`sam_utils.py`): `points_per_side=16`,
`pred_iou_thresh=0.8`, `stability_score_thresh=0.8`, `crop_n_layers=0`,
`crop_nms_thresh=0.4`, `points_per_batch=5`. Masks reaching a crop **corner**
(both a vertical and a horizontal edge) are rejected; a configurable fraction of
mask pixels within a band inside the arena ring is also rejected.

| Pipeline | Crop | SAM area (px) | Other |
|----------|------|---------------|-------|
| Heartbeat | 1048 | 8,000–35,000 | downscale 4; stationary epoch ≥29 s; endpoint search ≤5 frames; FastICA(5, `exp`, random_state=32, max_iter=1000, tol=1e-4); Alive threshold P≥0.50 |
| L-bend | 2048 | 25,000–200,000 | arena radius + 50 px margin zeroed; 4th-order midline polynomial (100 points) |
| GCaMP | 1048 | 8,000–35,000 | frames 0,7,14,22,29; arena from first frame reused; bright-brain if adjusted top-10% green ≥95 (crop 1048) |

---

## License

See `LICENSE` (add your chosen license before publishing).

## Citation

Please cite the accompanying paper if you use this code. (Add citation / BibTeX
entry here.)
