#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1-96

set -euo pipefail

mkdir -p logs

# ==========================================================
# REQUIRED VARIABLES PASSED FROM SUBMITTING SCRIPT
# ==========================================================

if [ -z "${PLATE_DIR:-}" ]; then
    echo "ERROR: PLATE_DIR is not set."
    exit 1
fi

if [ -z "${OUT_ROOT:-}" ]; then
    echo "ERROR: OUT_ROOT is not set."
    exit 1
fi

if [ -z "${PY_SCRIPT:-}" ]; then
    echo "ERROR: PY_SCRIPT is not set."
    exit 1
fi

if [ ! -d "$PLATE_DIR" ]; then
    echo "ERROR: PLATE_DIR does not exist:"
    echo "$PLATE_DIR"
    exit 1
fi

if [ ! -f "$PY_SCRIPT" ]; then
    echo "ERROR: PY_SCRIPT does not exist:"
    echo "$PY_SCRIPT"
    exit 1
fi

mkdir -p "$OUT_ROOT"

# ==========================================================
# USER CONFIG / PIPELINE PARAMETERS
# These can stay here unless you also want to pass them
# from the submitting script.
# ==========================================================

DEVICE="cpu"
MODEL_TYPE="vit_h"

# coarse motion / epoch detection
DOWNSCALE=4
MAX_FRAMES=0
FALLBACK_MIN_EPOCH_SEC=29.0

# fixed crop from detected ring center
CROP_SIZE=1048

# SAM search settings
LOWER_AREA=8000
UPPER_AREA=40000
SEARCH_MAX=5
PAD=0
DILATE_PX=0
MIN_SIZE=50

# epoch ICA fallback settings
ICA_N_COMPONENTS=5
ICA_RANDOM_STATE=32
ICA_FUN="exp"
ICA_MAX_ITER=1000
ICA_TOL=1e-4

# cProfile settings
PROFILE=1
PROFILE_TOP_N=200
REUSABLE_PARENT_DIR="${REUSABLE_PARENT_DIR:-}"

# ==========================================================
# Map array task id 1..96 to well name A1..H12
# ==========================================================

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is empty."
    exit 1
fi

WELLS=()
for R in A B C D E F G H; do
  for C in {1..12}; do
    WELLS+=("${R}${C}")
  done
done

IDX=$((SLURM_ARRAY_TASK_ID - 1))

if [ "$IDX" -lt 0 ] || [ "$IDX" -ge "${#WELLS[@]}" ]; then
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

WELL=${WELLS[$IDX]}

echo "=========================================================="
echo "Heartbeat extractor"
echo "Job ID       : ${SLURM_JOB_ID:-NA}"
echo "Array task   : ${SLURM_ARRAY_TASK_ID}"
echo "Well         : $WELL"
echo "Plate dir    : $PLATE_DIR"
echo "Output root  : $OUT_ROOT"
echo "Reusable dir : ${REUSABLE_PARENT_DIR:-<disabled>}"
echo "Python script: $PY_SCRIPT"
echo "Host         : $(hostname)"
echo "Start time   : $(date)"
echo "=========================================================="

python "$PY_SCRIPT" \
  --plate_dir "$PLATE_DIR" \
  --out_root "$OUT_ROOT" \
  --well "$WELL" \
  --downscale "$DOWNSCALE" \
  --max_frames "$MAX_FRAMES" \
  --fallback_min_epoch_sec "$FALLBACK_MIN_EPOCH_SEC" \
  --crop_size "$CROP_SIZE" \
  --device "$DEVICE" \
  --model_type "$MODEL_TYPE" \
  --lower_area "$LOWER_AREA" \
  --upper_area "$UPPER_AREA" \
  --search_max "$SEARCH_MAX" \
  --pad "$PAD" \
  --dilate_px "$DILATE_PX" \
  --min_size "$MIN_SIZE" \
  --ica_n_components "$ICA_N_COMPONENTS" \
  --ica_random_state "$ICA_RANDOM_STATE" \
  --ica_fun "$ICA_FUN" \
  --ica_max_iter "$ICA_MAX_ITER" \
    --ica_tol "$ICA_TOL" \
    --profile "$PROFILE" \
    --profile_top_n "$PROFILE_TOP_N" \
    --reusable_parent_dir "$REUSABLE_PARENT_DIR"

echo "=========================================================="
echo "Finished well: $WELL"
echo "End time     : $(date)"
echo "=========================================================="