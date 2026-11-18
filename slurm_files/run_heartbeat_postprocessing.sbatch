#!/bin/bash
#SBATCH --job-name=hb_montage
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# ==========================================================
# Check inputs
# ==========================================================

: "${ROOT_DIR:?ROOT_DIR is required}"
: "${MONTAGE_SCRIPT:?MONTAGE_SCRIPT is required}"

if [ ! -d "$ROOT_DIR" ]; then
    echo "ERROR: ROOT_DIR does not exist:"
    echo "$ROOT_DIR"
    exit 1
fi

if [ ! -f "$MONTAGE_SCRIPT" ]; then
    echo "ERROR: MONTAGE_SCRIPT does not exist:"
    echo "$MONTAGE_SCRIPT"
    exit 1
fi

echo "=========================================================="
echo "Heartbeat montage job"
echo "Job ID        : ${SLURM_JOB_ID:-NA}"
echo "Host          : $(hostname)"
echo "Start time    : $(date)"
echo "ROOT_DIR      : $ROOT_DIR"
echo "MONTAGE_SCRIPT: $MONTAGE_SCRIPT"
echo "=========================================================="

# ==========================================================
# Run montage script
# ==========================================================

MONTAGE_PROFILE=1
MONTAGE_PROFILE_TOP_N=200

if [ "$MONTAGE_PROFILE" -eq 1 ]; then
    PROFILE_DIR="$ROOT_DIR/montages/profiling"
    PROFILE_PROF="$PROFILE_DIR/montage.cprofile.prof"
    PROFILE_TXT="$PROFILE_DIR/montage.cprofile.txt"

    mkdir -p "$PROFILE_DIR"

    python -m cProfile -o "$PROFILE_PROF" "$MONTAGE_SCRIPT" "$ROOT_DIR"

    python - "$PROFILE_PROF" "$PROFILE_TXT" "$MONTAGE_PROFILE_TOP_N" <<'PY'
import sys
import pstats

prof_path = sys.argv[1]
txt_path = sys.argv[2]
top_n = int(sys.argv[3])

with open(txt_path, "w") as f:
    f.write(f"profile_binary\t{prof_path}\n")
    f.write("\nTOP_FUNCTIONS_BY_CUMULATIVE_TIME\n")
    stats_cum = pstats.Stats(prof_path, stream=f)
    stats_cum.sort_stats("cumulative")
    stats_cum.print_stats(top_n)

    f.write("\nTOP_FUNCTIONS_BY_SELF_TIME\n")
    stats_tot = pstats.Stats(prof_path, stream=f)
    stats_tot.sort_stats("tottime")
    stats_tot.print_stats(top_n)

print(f"[PROFILE] Saved montage cProfile binary: {prof_path}")
print(f"[PROFILE] Saved montage cProfile report: {txt_path}")
PY
else
    python "$MONTAGE_SCRIPT" "$ROOT_DIR"
fi

echo "=========================================================="
echo "Finished heartbeat montage job"
echo "End time: $(date)"
echo "=========================================================="