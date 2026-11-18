#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=lbend_montage
#SBATCH --output=logs4/%x_%A_%a.out
#SBATCH --error=logs4/%x_%A_%a.err

set -euo pipefail

mkdir -p "$LOG_DIR"

# ----------------------------------------------------------
# Required variables passed by submit_lbend_pipeline.sh
# ----------------------------------------------------------
: "${ROOT_DIR:?ROOT_DIR is required}"
: "${MONTAGE_SCRIPT:?MONTAGE_SCRIPT is required}"
: "${LOG_DIR:?LOG_DIR is required}"

echo "======================================"
echo "Running L-bend postprocessing"
echo "SLURM job ID    : ${SLURM_JOB_ID}"
echo "ROOT_DIR        : ${ROOT_DIR}"
echo "MONTAGE_SCRIPT  : ${MONTAGE_SCRIPT}"
echo "LOG_DIR         : ${LOG_DIR}"
echo "======================================"

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "ERROR: ROOT_DIR does not exist:"
    echo "$ROOT_DIR"
    exit 1
fi

if [[ ! -f "$MONTAGE_SCRIPT" ]]; then
    echo "ERROR: MONTAGE_SCRIPT does not exist:"
    echo "$MONTAGE_SCRIPT"
    exit 1
fi

python "$MONTAGE_SCRIPT" \
    --root_path "$ROOT_DIR" \
    --thumb_w 500 \
    --thumb_h 500 \
    --border_px 4

echo "✅ Finished L-bend postprocessing"
echo "Montages should be in:"
echo "${ROOT_DIR}/montages"