#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs4/%x_%j.out
#SBATCH --error=logs4/%x_%j.err
#SBATCH --job-name=tracking_summary

set -euo pipefail

: "${INPUT_DIR:?INPUT_DIR is required}"
: "${OUT_DIR:?OUT_DIR is required}"
: "${PY_SCRIPT:?PY_SCRIPT is required}"

mkdir -p "$OUT_DIR"

python "$PY_SCRIPT" \
    --root_dir  "$INPUT_DIR" \
    --output_dir "$OUT_DIR" 