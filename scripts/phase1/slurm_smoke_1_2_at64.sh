#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- SMOKE TEST for the 1/2 @ input-64 (fm32) arm
# =============================================================================
# Fresh 1-epoch run on the gh-dev partition (30 min cap) with a throwaway
# experiment name.  Purpose: confirm, before committing the multi-day `gh` job,
# that the CHECKPOINTED fm32 correlation at batch 4 x accum 2 fits in ~96 GB,
# that loss is finite, and the per-epoch time is sane.  (This config was already
# verified on an idev node 2026-07-09: ~66 GB, 4.0 samp/s, no OOM -- so this
# sbatch smoke is optional insurance if you skipped the idev check.)
#
# The real run resumes the copied 5090 checkpoint; this smoke does NOT resume
# (it starts fresh so epoch 0 completes -- and validates -- inside 30 min).
#
# PASS criteria (see runbook):
#   - "cuda True" banner, params printed, no OutOfMemoryError
#   - Loss lines are finite (no nan)
#   - ~500 optimiser steps finish and a "Validation ... EPE:" line prints
#
# If it OOMs (only if the GPU is not clean): fall back to batch 1 x accum 8
# non-checkpointed (edit both this and the real script) -- both keep the same
# eff batch 8 and total_steps 75000, so resume stays valid either way.
#
# Usage:  sbatch scripts/phase1/slurm_smoke_1_2_at64.sh
# =============================================================================
#SBATCH --job-name=smk12at64
#SBATCH --output=logs/smoke_1_2_at64_%j.out
# no --error: SLURM merges stderr (Python logging INFO lines) into the .out file
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# NOTE: no --gres on Vista; the gh-dev partition allocates a whole GH200 node.
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail

# Vista has no `conda` module; conda lives under $WORK/miniconda3 (set up in
# .bashrc, which batch jobs do not source -- so source it by absolute path).
source "$WORK/miniconda3/etc/profile.d/conda.sh"
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

srun python -u scripts/phase1/train_phase1.py \
    --model-config configs/models/raft_dvc_1_2_p2_r4_ckpt.yaml \
    --data-config  r4_medium_size64 \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name _smoke_1_2_at64 \
    --epochs 1 --batch-size 4 --grad-accum-steps 2 \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed 42

echo "=== SMOKE done @ $(date) -- if you saw an EPE line and no OOM, submit the real job ==="
