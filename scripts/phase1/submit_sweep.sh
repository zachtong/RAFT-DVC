#!/bin/bash
# =============================================================================
# Convenience wrapper: submit the Phase-1 OTF sweep array job.
# =============================================================================
# Usage:
#     bash scripts/phase1/submit_sweep.sh                 # default 4 concurrent
#     CONCURRENT=8 bash scripts/phase1/submit_sweep.sh    # 8 concurrent
#     INDICES=0-9   bash scripts/phase1/submit_sweep.sh   # only size16 (rows 0-9)
#     INDICES=10-19 bash scripts/phase1/submit_sweep.sh   # only size32 (rows 10-19)
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/projects/RAFT-DVC}"
cd "$PROJECT_DIR"

# Sanity: case_matrix + script exist
[ -f configs/phase1/case_matrix.yaml ] || { echo "MISSING configs/phase1/case_matrix.yaml"; exit 1; }
[ -f scripts/phase1/slurm_train_otf.sh ] || { echo "MISSING slurm_train_otf.sh"; exit 1; }

# Defaults
CONCURRENT="${CONCURRENT:-4}"
INDICES="${INDICES:-}"

if [ -z "$INDICES" ]; then
    N_CASES=$(python scripts/phase1/case_helper.py \
        --matrix configs/phase1/case_matrix.yaml --count)
    LAST=$((N_CASES - 1))
    INDICES="0-${LAST}"
fi

ARRAY_SPEC="${INDICES}%${CONCURRENT}"

echo "===================================================================="
echo "Phase-1 OTF sweep submission"
echo "Project dir   : $PROJECT_DIR"
echo "Array spec    : $ARRAY_SPEC"
echo "Cases preview :"
python scripts/phase1/case_helper.py \
    --matrix configs/phase1/case_matrix.yaml --list | head -30
echo "===================================================================="

read -rp "Proceed? [y/N] " ans
case "$ans" in
    y|Y) ;;
    *) echo "Aborted."; exit 0 ;;
esac

sbatch --array="${ARRAY_SPEC}" scripts/phase1/slurm_train_otf.sh
echo "Submitted. Use 'squeue -u \$USER' to monitor."
