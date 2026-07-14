#!/bin/bash
# =============================================================================
# TACC Vista -- regenerate the r8_medium_size128 dataset on $SCRATCH.
# The generator is config-hash seeded and deterministic, so regenerating on
# Vista reproduces the local dataset without a 94 GB upload.
# Usage:  sbatch scripts/paper1/slurm_gen_r8md128_vista.sh
# =============================================================================
#SBATCH --job-name=gen-r8md128
#SBATCH --output=logs/gen_r8md128_%j.out
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/envs/raft-dvc"

PROJECT_DIR="$WORK/projects/RAFT-DVC"
DATA_ROOT="$SCRATCH/raft-dvc/data_paper1_axis2"
cd "$PROJECT_DIR"
mkdir -p logs "$DATA_ROOT"

if [ -d "$DATA_ROOT/r8_medium_size128/train" ] && \
   [ "$(ls "$DATA_ROOT/r8_medium_size128/train" | wc -l)" -ge 2000 ]; then
    echo "[done] dataset already present -- exiting."
    exit 0
fi

srun python -u scripts/generate_phase1_dataset.py \
    --only-config r8_medium --sizes 128 --radii 8 \
    --density-per-1000 0.075 --input-disp-range 8 16 \
    --train 2000 --val 200 --test 200 \
    --output-root "$DATA_ROOT" --no-mixed --no-viz

echo "=== generation finished @ $(date) ==="
ls "$DATA_ROOT/r8_medium_size128" && ls "$DATA_ROOT/r8_medium_size128/train" | wc -l
