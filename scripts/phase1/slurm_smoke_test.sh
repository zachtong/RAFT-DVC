#!/bin/bash
# =============================================================================
# SLURM smoke test: ONE case, 10 epoch, gh-dev partition (30 min walltime).
# =============================================================================
# Goal: validate the env + OTF + per-GH200-node performance before submitting
# the full 20-case sweep.
#
# Submit:
#     sbatch scripts/phase1/slurm_smoke_test.sh
# Monitor:
#     squeue -u $USER
#     tail -f logs/raft_smoke_<jobid>.out
#
# Success criteria (check the log):
#   - "CUDA available : True"
#   - "Training" starts and ~30-90 sec/epoch on GH200 batch=24
#   - val/epe at epoch 9 < 1.0 (sane convergence start)
#   - no OOM, no NaN
# =============================================================================

#SBATCH --job-name=raft-smoke
#SBATCH --output=logs/raft_smoke_%j.out
#SBATCH --error=logs/raft_smoke_%j.err
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
# Note: gh-dev partition implies 1 GH200 per node -- no --gres needed.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/projects/RAFT-DVC}"
cd "$PROJECT_DIR"
mkdir -p logs

# Source conda init -- TACC Vista has no `conda` module; use user's miniconda.
# shellcheck disable=SC1091
if [ -f "$WORK/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$WORK/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: cannot find conda init script" >&2
    exit 2
fi
conda activate "$WORK/envs/raft-dvc"

export CHECKPOINTS_ROOT="$WORK/checkpoints/raft-dvc"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$CHECKPOINTS_ROOT/phase1"

echo "===================================================================="
echo "SMOKE TEST: 1/2 encoder + r4_medium_size32 + OTF + 10 epoch"
echo "Node      : $SLURM_NODELIST"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time      : $(date)"
echo "===================================================================="

python scripts/phase1/train_phase1.py \
    --model-config configs/models/raft_dvc_1_2_p2_r4.yaml \
    --data-config  r4_medium_size32 \
    --experiment-name smoke_gh200_otf \
    --epochs 10 \
    --batch-size 24 \
    --num-workers 32 \
    --max-lr 4.0e-4 \
    --otf \
    --otf-train-length 1200 \
    --otf-val-length 96

echo "Smoke test done at $(date)"
