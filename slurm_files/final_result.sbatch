#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs4/%x_%j.out
#SBATCH --error=logs4/%x_%j.err
#SBATCH --job-name=join_final_results

set -euo pipefail

: "${ROOT_DIR:?ROOT_DIR is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${PY_SCRIPT:?PY_SCRIPT is required}"

mkdir -p "$OUTPUT_DIR"

# Optional override; default preserves the original timepoints.
TIMEPOINTS="${TIMEPOINTS:-30 45 60 120 180}"

python "$PY_SCRIPT" \
    --root_dir "$ROOT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --timepoints $TIMEPOINTS