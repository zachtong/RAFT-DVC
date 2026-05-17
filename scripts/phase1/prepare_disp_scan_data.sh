#!/bin/bash
# =============================================================================
# Paper-2 displacement scan: pre-generate all 20 NPZ datasets on TACC.
# =============================================================================
# Generates training + test data for each (config, disp_range) pair in
# configs/phase1/case_matrix_disp_scan.yaml.
#
# Output layout (on $SCRATCH):
#   data_paper2_disp_scan/r4_medium_size64_d0.2/{train,val,test}/...
#   data_paper2_disp_scan/r4_medium_size64_d0.6/...
#   ... etc (20 dirs total)
#
# Generation is CPU-only -- runs on login node in ~30-60 min total.
# Run with:
#   bash scripts/phase1/prepare_disp_scan_data.sh
#
# Skips configs that already exist (idempotent).
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/projects/RAFT-DVC}"
cd "$PROJECT_DIR"

# Activate conda env (needed for PyYAML, numpy, etc.)
# shellcheck disable=SC1091
if [ -f "$WORK/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$WORK/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$WORK/envs/raft-dvc"

OUT_ROOT="$SCRATCH/raft-dvc/data_paper2_disp_scan"
mkdir -p "$OUT_ROOT"

# Use case_helper to list all 20 cases with their input_disp ranges
N_CASES=$(python scripts/phase1/case_helper.py \
    --matrix configs/phase1/case_matrix_disp_scan.yaml --count)
echo "Preparing $N_CASES paper-2 disp-scan datasets at $OUT_ROOT"
echo "===================================================================="

for ((i=0; i<N_CASES; i++)); do
    eval "$(python scripts/phase1/case_helper.py \
        --matrix configs/phase1/case_matrix_disp_scan.yaml --index "$i")"

    DISP_TAG="d$(printf '%g' "$INPUT_DISP_MAX")"
    OUT_DIR="$OUT_ROOT/${DATA}_size64_${DISP_TAG}"

    if [ -d "$OUT_DIR/train" ] && [ -d "$OUT_DIR/test" ]; then
        echo "[$i/$N_CASES] $NAME -> already exists, skipping"
        continue
    fi

    echo "[$i/$N_CASES] $NAME -> generating in $OUT_DIR"
    # Use a temp output dir so generator's naming doesn't collide,
    # then move into our disp-tagged dir.
    TMP_ROOT="$OUT_ROOT/_tmp_$NAME"
    rm -rf "$TMP_ROOT"

    python scripts/generate_phase1_dataset.py \
        --sizes 64 \
        --train 1200 --val 100 --test 100 \
        --no-mixed --no-viz \
        --only-config "$DATA" \
        --input-disp-range "$INPUT_DISP_MIN" "$INPUT_DISP_MAX" \
        --output-root "$TMP_ROOT"

    # Generator outputs to <root>/<DATA>_size64/; rename to disp-tagged dir
    mv "$TMP_ROOT/${DATA}_size64" "$OUT_DIR"
    rm -rf "$TMP_ROOT"
done

echo "===================================================================="
echo "All paper-2 disp-scan datasets ready at $OUT_ROOT"
ls "$OUT_ROOT" | head -25
