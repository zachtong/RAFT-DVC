#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- SMOKE TEST for the 1/2 @ input-64 (fm32) arm
# =============================================================================
# Fresh 1-epoch run on the gh-dev partition (30 min cap) with a throwaway
# experiment name.  Purpose: confirm, before committing the multi-day `gh` job,
# that batch 2 of the fm32 correlation fits in 96 GB WITHOUT gradient
# checkpointing, that loss is finite, and the per-epoch time is sane.
#
# The real run resumes the copied 5090 checkpoint; this smoke does NOT resume
# (it starts fresh so epoch 0 completes -- and validates -- inside 30 min).
#
# PASS criteria (see runbook):
#   - "cuda True" banner, params printed, no OutOfMemoryError
#   - Loss lines are finite (no nan)
#   - ~1000 optimiser steps finish and a "Validation ... EPE:" line prints
#
# If it OOMs: fall back to batch 1 x accum 8 (edit both this and the real
# script) -- batch 1 fm32 non-checkpointed is ~31 GB and always fits.
#
# Usage:  sbatch scripts/phase1/slurm_smoke_1_2_at64.sh
# =============================================================================
#SBATCH --job-name=smk12at64
#SBATCH --output=logs/smoke_1_2_at64_%j.out
#SBATCH --error=logs/smoke_1_2_at64_%j.err
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/raft-dvc"

PROJECT_DIR="$WORK/projects/RAFT-DVC"
DATA_ROOT="$SCRATCH/raft-dvc/data_paper1_v3"
OUT_ROOT="$SCRATCH/raft-dvc/training_runs"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT/phase1"

echo "=== SMOKE Job $SLURM_JOB_ID on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0))"
[ -d "$DATA_ROOT/r4_medium_size64/train" ] || { echo "ERROR: data missing at $DATA_ROOT/r4_medium_size64 -- see runbook step 3"; exit 1; }

srun python scripts/phase1/train_phase1.py \
    --model-config configs/models/raft_dvc_1_2_p2_r4.yaml \
    --data-config  r4_medium_size64 \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name _smoke_1_2_at64 \
    --epochs 1 --batch-size 2 --grad-accum-steps 4 \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed 42

echo "=== SMOKE done @ $(date) -- if you saw an EPE line and no OOM, submit the real job ==="
