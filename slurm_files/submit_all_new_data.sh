#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ==========================================================
# Collated launcher: runs the 4 post-processing pipelines
# that were previously submitted separately:
#   1. Heartbeat feature-matrix + montage
#   2. L-bend severity + montage
#   3. GCaMP fluorescence + montage
#   4. Tracking summary
#
# Input and output roots are shared across all four.
#
# Run with bash, NOT sbatch:
#   bash submit_all_new_data.sh --root /path/to/plate_root
# ==========================================================

# ==========================================================
# USER INPUT
# ==========================================================
ROOT_INPUT_DIR="${ROOT_INPUT_DIR:-}"

# Optional CLI override: --root / -r
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--root)
            if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
                echo "ERROR: $1 requires an input directory." >&2
                exit 1
            fi
            ROOT_INPUT_DIR="$2"; shift 2;;
        -h|--help)
            echo "Usage: bash $(basename "$0") --root ROOT_INPUT_DIR"
            echo "Alternatively, set ROOT_INPUT_DIR in the environment."
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "$ROOT_INPUT_DIR" ]]; then
    echo "ERROR: provide an input directory with --root or ROOT_INPUT_DIR." >&2
    exit 1
fi

if [[ ! -d "$ROOT_INPUT_DIR" ]]; then
    echo "ERROR: ROOT_INPUT_DIR does not exist: $ROOT_INPUT_DIR" >&2
    exit 1
fi

ROOT_INPUT_DIR="$(cd -- "$ROOT_INPUT_DIR" && pwd)"
OUT_ROOT="${ROOT_INPUT_DIR}/gittest"
ROOT_SET_NAME="$(basename "$ROOT_INPUT_DIR")"
REUSABLE_PARENT_DIR="${ROOT_INPUT_DIR}/reusable_cache/${ROOT_SET_NAME}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-${ROOT_INPUT_DIR}/final_results}"

# ==========================================================
# Pipeline scripts / sbatch files
# ==========================================================
# Heartbeat
HB_PY_SCRIPT="$RELEASE_ROOT/sam_pipeline/heartbeat_pipeline_xgboost_CLEAN.py"
HB_MONTAGE_SCRIPT="$RELEASE_ROOT/sam_pipeline/make_heartbeat_montages_FINAL_CLEAN.py"
HB_SBATCH="$SCRIPT_DIR/mobility_hb.sbatch"
HB_MONTAGE_SBATCH="$SCRIPT_DIR/run_heartbeat_postprocessing.sbatch"

# L-bend
LB_PY_SCRIPT="$RELEASE_ROOT/sam_pipeline/lbend_seveirity.py"
LB_MONTAGE_SCRIPT="$RELEASE_ROOT/sam_pipeline/make_lbend_montages.py"
LB_SBATCH="$SCRIPT_DIR/l_bend.sbatch"
LB_MONTAGE_SBATCH="$SCRIPT_DIR/run_lbend_postprocessing.sbatch"

# GCaMP fluorescence
GC_PY_SCRIPT="$RELEASE_ROOT/sam_pipeline/g_camp.py"
GC_MONTAGE_SCRIPT="$RELEASE_ROOT/sam_pipeline/montage_adjusted_bright_brain.py"
GC_SBATCH="$SCRIPT_DIR/g_camp.sbatch"
GC_MONTAGE_SBATCH="$SCRIPT_DIR/run_gcamp_montage.sbatch"

# Tracking summary
TRACKING_PY_SCRIPT="$RELEASE_ROOT/sam_pipeline/tracking_all_tp.py"
TRACKING_SBATCH="$SCRIPT_DIR/run_tracking_summary.sbatch"

# Final aggregation (runs after all 4 extractors finish)
FINAL_PY_SCRIPT="$RELEASE_ROOT/sam_pipeline/final_join_with_adjusted_fluoro.py"
FINAL_SBATCH="$SCRIPT_DIR/final_result.sbatch"

# ==========================================================
# Safety checks
# ==========================================================
if [ -n "${SLURM_JOB_ID-}" ]; then
    echo "ERROR: run this launcher with 'bash $(basename "$0")', not with sbatch." >&2
    exit 1
fi

for f in \
    "$HB_PY_SCRIPT" "$HB_MONTAGE_SCRIPT" "$HB_SBATCH" "$HB_MONTAGE_SBATCH" \
    "$LB_PY_SCRIPT" "$LB_MONTAGE_SCRIPT" "$LB_SBATCH" "$LB_MONTAGE_SBATCH" \
    "$GC_PY_SCRIPT" "$GC_MONTAGE_SCRIPT" "$GC_SBATCH" "$GC_MONTAGE_SBATCH" \
    "$TRACKING_PY_SCRIPT" "$TRACKING_SBATCH" "$FINAL_PY_SCRIPT" "$FINAL_SBATCH"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file missing: $f"
        exit 1
    fi
done

# Collect every submitted job ID so the final step can depend on all of them
ALL_JOBIDS=()

mkdir -p "$OUT_ROOT"
mkdir -p "$REUSABLE_PARENT_DIR"
mkdir -p logs logs4

echo "=========================================================="
echo "Collated pipeline launcher"
echo "ROOT_INPUT_DIR : $ROOT_INPUT_DIR"
echo "OUT_ROOT       : $OUT_ROOT"
echo "REUSABLE_PARENT: $REUSABLE_PARENT_DIR"
echo "=========================================================="

# ==========================================================
# 1. Heartbeat feature-matrix + montage
# ==========================================================
mapfile -t HR_FOLDERS < <(
    find "$ROOT_INPUT_DIR" -maxdepth 1 -mindepth 1 -type d \
        \( -iname "*_HR_*" -o -iname "*_HB_*" \) | sort
)
echo ""
echo "### Heartbeat: found ${#HR_FOLDERS[@]} HR/HB folder(s)"
for PLATE_DIR in "${HR_FOLDERS[@]:-}"; do
    [ -z "${PLATE_DIR:-}" ] && continue
    PLATE_NAME=$(basename "$PLATE_DIR")
    ROOT_DIR="${OUT_ROOT}/${PLATE_NAME}"
    LOG_DIR="${ROOT_DIR}/logs"
    mkdir -p "$ROOT_DIR" "$LOG_DIR"

    ARRAY_JOBID=$(sbatch --parsable \
        --export=ALL,PLATE_DIR="$PLATE_DIR",OUT_ROOT="$OUT_ROOT",PY_SCRIPT="$HB_PY_SCRIPT",LOG_DIR="$LOG_DIR",REUSABLE_PARENT_DIR="$REUSABLE_PARENT_DIR" \
        "$HB_SBATCH")
    echo "  [$PLATE_NAME] heartbeat array job: $ARRAY_JOBID"
    ALL_JOBIDS+=("$ARRAY_JOBID")

    MONTAGE_JOBID=$(sbatch --parsable \
        --dependency=afterany:${ARRAY_JOBID} \
        --export=ALL,ROOT_DIR="$ROOT_DIR",MONTAGE_SCRIPT="$HB_MONTAGE_SCRIPT",LOG_DIR="$LOG_DIR" \
        "$HB_MONTAGE_SBATCH")
    echo "  [$PLATE_NAME] heartbeat montage job: $MONTAGE_JOBID (after $ARRAY_JOBID)"
    ALL_JOBIDS+=("$MONTAGE_JOBID")
done

# ==========================================================
# 2. L-bend severity + montage
# ==========================================================
mapfile -t MORPH_FOLDERS < <(
    find "$ROOT_INPUT_DIR" -maxdepth 1 -mindepth 1 -type d \
        -iname "*morphology*" | sort
)
echo ""
echo "### L-bend: found ${#MORPH_FOLDERS[@]} morphology folder(s)"
for PLATE_DIR in "${MORPH_FOLDERS[@]:-}"; do
    [ -z "${PLATE_DIR:-}" ] && continue
    PLATE_NAME=$(basename "$PLATE_DIR")
    ROOT_DIR="${OUT_ROOT}/${PLATE_NAME}"
    LOG_DIR="${ROOT_DIR}/logs"
    mkdir -p "$ROOT_DIR" "$LOG_DIR"

    ARRAY_JOBID=$(sbatch --parsable \
        --export=ALL,PLATE_DIR="$PLATE_DIR",ROOT_DIR="$ROOT_DIR",PY_SCRIPT="$LB_PY_SCRIPT",LOG_DIR="$LOG_DIR",REUSABLE_PARENT_DIR="$REUSABLE_PARENT_DIR" \
        "$LB_SBATCH")
    echo "  [$PLATE_NAME] L-bend array job: $ARRAY_JOBID"
    ALL_JOBIDS+=("$ARRAY_JOBID")

    MONTAGE_JOBID=$(sbatch --parsable \
        --dependency=afterany:${ARRAY_JOBID} \
        --export=ALL,ROOT_DIR="$ROOT_DIR",MONTAGE_SCRIPT="$LB_MONTAGE_SCRIPT",LOG_DIR="$LOG_DIR" \
        "$LB_MONTAGE_SBATCH")
    echo "  [$PLATE_NAME] L-bend montage job: $MONTAGE_JOBID (after $ARRAY_JOBID)"
    ALL_JOBIDS+=("$MONTAGE_JOBID")
done

# ==========================================================
# 3. GCaMP fluorescence + montage
# ==========================================================
mapfile -t FLUORO_FOLDERS < <(
    find "$ROOT_INPUT_DIR" -maxdepth 1 -mindepth 1 -type d \
        -iname "*fluor*" | sort
)
echo ""
echo "### GCaMP: found ${#FLUORO_FOLDERS[@]} fluoro folder(s)"
for PLATE_DIR in "${FLUORO_FOLDERS[@]:-}"; do
    [ -z "${PLATE_DIR:-}" ] && continue
    PLATE_NAME=$(basename "$PLATE_DIR")
    ROOT_DIR="${OUT_ROOT}/${PLATE_NAME}_per_well_fluorescence"
    LOG_DIR="${ROOT_DIR}/logs"
    mkdir -p "$ROOT_DIR" "$LOG_DIR"

    ARRAY_JOBID=$(sbatch --parsable \
        --export=ALL,PLATE_DIR="$PLATE_DIR",OUT_ROOT="$OUT_ROOT",PY_SCRIPT="$GC_PY_SCRIPT",LOG_DIR="$LOG_DIR",REUSABLE_PARENT_DIR="$REUSABLE_PARENT_DIR" \
        "$GC_SBATCH")
    echo "  [$PLATE_NAME] fluorescence array job: $ARRAY_JOBID"
    ALL_JOBIDS+=("$ARRAY_JOBID")

    MONTAGE_JOBID=$(sbatch --parsable \
        --dependency=afterany:${ARRAY_JOBID} \
        --export=ALL,ROOT_DIR="$ROOT_DIR",MONTAGE_SCRIPT="$GC_MONTAGE_SCRIPT",LOG_DIR="$LOG_DIR" \
        "$GC_MONTAGE_SBATCH")
    echo "  [$PLATE_NAME] fluorescence montage job: $MONTAGE_JOBID (after $ARRAY_JOBID)"
    ALL_JOBIDS+=("$MONTAGE_JOBID")
done

# ==========================================================
# 4. Tracking summary
# ==========================================================
echo ""
echo "### Tracking summary"
TRACKING_JOBID=$(sbatch --parsable \
    --export=ALL,INPUT_DIR="$ROOT_INPUT_DIR",OUT_DIR="$OUT_ROOT",PY_SCRIPT="$TRACKING_PY_SCRIPT" \
    "$TRACKING_SBATCH")
echo "  tracking summary job: $TRACKING_JOBID"
ALL_JOBIDS+=("$TRACKING_JOBID")

# ==========================================================
# 5. Final aggregation (runs after ALL extractors complete)
# ==========================================================
echo ""
echo "### Final aggregation"
if [ ${#ALL_JOBIDS[@]} -eq 0 ]; then
    echo "ERROR: no extractor jobs were submitted; skipping final aggregation."
    exit 1
fi

# Deduplicate and build the afterany dependency list
IFS=$'\n' UNIQ_JOBIDS=($(printf '%s\n' "${ALL_JOBIDS[@]}" | awk '!x[$0]++')); unset IFS
DEPEND=$(IFS=, ; echo "${UNIQ_JOBIDS[*]}")

FINAL_JOBID=$(sbatch --parsable \
    --dependency=afterany:${DEPEND} \
    --export=ALL,ROOT_DIR="$OUT_ROOT",OUTPUT_DIR="$FINAL_OUTPUT_DIR",PY_SCRIPT="$FINAL_PY_SCRIPT" \
    "$FINAL_SBATCH")
echo "  final aggregation job: $FINAL_JOBID"
echo "  depends on: $DEPEND"
echo "  final output dir: $FINAL_OUTPUT_DIR"

echo ""
echo "=========================================================="
echo "All pipelines submitted."
echo "Check jobs with: squeue -u $USER"
echo "=========================================================="
