#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=1-96
#SBATCH --output=logs/fluor_%A_%a.out
#SBATCH --error=logs/fluor_%A_%a.err

set -euo pipefail

mkdir -p logs

# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------
PLATE_DIR="${PLATE_DIR:?PLATE_DIR not set}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT not set}"
PY_SCRIPT="${PY_SCRIPT:?PY_SCRIPT not set}"
REUSABLE_PARENT_DIR="${REUSABLE_PARENT_DIR:-}"

# ------------------------------------------------------------
# Convert Slurm array task ID 1-96 to A1-H12
# ------------------------------------------------------------
ROWS=(A B C D E F G H)

TASK_ID=${SLURM_ARRAY_TASK_ID}

ROW_INDEX=$(( (TASK_ID - 1) / 12 ))
COL_INDEX=$(( (TASK_ID - 1) % 12 + 1 ))

ROW=${ROWS[$ROW_INDEX]}
WELL="${ROW}${COL_INDEX}"

echo "=========================================="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing well: ${WELL}"
echo "Plate dir: ${PLATE_DIR}"
echo "Output root: ${OUT_ROOT}"
echo "Python script: ${PY_SCRIPT}"
echo "=========================================="

python "$PY_SCRIPT" \
    --plate_dir "$PLATE_DIR" \
    --out_root "$OUT_ROOT" \
    --well "$WELL" \
    --target_frames "0,7,14,22,29" \
    --device cpu \
    --reusable_parent_dir "$REUSABLE_PARENT_DIR"
    # --crop_size 1048 \
    # --lower_area 8000 \
    # --upper_area 30000