#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/montage_%j.out
#SBATCH --error=logs/montage_%j.err

set -euo pipefail

mkdir -p logs

ROOT_DIR="${ROOT_DIR:?ROOT_DIR not set}"
MONTAGE_SCRIPT="${MONTAGE_SCRIPT:?MONTAGE_SCRIPT not set}"

echo "=========================================="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "ROOT_DIR: ${ROOT_DIR}"
echo "MONTAGE_SCRIPT: ${MONTAGE_SCRIPT}"
echo "=========================================="

python "$MONTAGE_SCRIPT" \
    --root_dir "$ROOT_DIR"