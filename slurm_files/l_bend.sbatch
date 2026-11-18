#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=1-96
#SBATCH --job-name=lbend_96
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "$LOG_DIR"
mkdir -p "$ROOT_DIR"

# ----------------------------------------------------------
# Required variables passed by submit_lbend_pipeline.sh
# ----------------------------------------------------------
: "${PLATE_DIR:?PLATE_DIR is required}"
: "${ROOT_DIR:?ROOT_DIR is required}"
: "${PY_SCRIPT:?PY_SCRIPT is required}"
: "${LOG_DIR:?LOG_DIR is required}"
REUSABLE_PARENT_DIR="${REUSABLE_PARENT_DIR:-}"
MIN_RING_RADIUS="${MIN_RING_RADIUS:-800}"

# ----------------------------------------------------------
# WELL MAP: 1-96 -> A1-H12
# ----------------------------------------------------------
ROWS=(A B C D E F G H)

TASK_ID=${SLURM_ARRAY_TASK_ID}

ROW_INDEX=$(( (TASK_ID - 1) / 12 ))
COL_INDEX=$(( (TASK_ID - 1) % 12 + 1 ))

ROW=${ROWS[$ROW_INDEX]}
WELL="${ROW}${COL_INDEX}"

TIF_PATH="${PLATE_DIR}/well_${WELL}.tif"

echo "======================================"
echo "SLURM job ID  : ${SLURM_JOB_ID}"
echo "Array job ID  : ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID : ${TASK_ID}"
echo "Processing well: ${WELL}"
echo "TIF path      : ${TIF_PATH}"
echo "ROOT_DIR      : ${ROOT_DIR}"
echo "PY_SCRIPT     : ${PY_SCRIPT}"
echo "LOG_DIR       : ${LOG_DIR}"
echo "Min ring radius: ${MIN_RING_RADIUS} px"
echo "======================================"

if [[ ! -f "$TIF_PATH" ]]; then
    echo "ERROR: TIF file not found:"
    echo "$TIF_PATH"
    exit 1
fi

python "$PY_SCRIPT" \
    --tif_path "$TIF_PATH" \
    --output_root "$ROOT_DIR" \
    --crop_size 2048 \
    --min_ring_radius "$MIN_RING_RADIUS" \
    --lower_area 25000 \
    --upper_area 200000 \
    --device cpu \
    --reusable_parent_dir "$REUSABLE_PARENT_DIR"

echo "✅ Finished well ${WELL}"