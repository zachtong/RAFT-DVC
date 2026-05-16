#!/bin/bash
# =============================================================================
# SLURM array job: Phase-1 OTF sweep on TACC Vista GH200
# =============================================================================
# Each array task trains one case from configs/phase1/case_matrix.yaml.
# Submit with:
#     sbatch --array=0-19%4 scripts/phase1/slurm_train_otf.sh
# (4 concurrent tasks; 20 total -> wall-clock ~ ceil(20/4) * (per-case hours).)
#
# Resume only failed tasks:
#     sbatch --array=3,7,12%4 scripts/phase1/slurm_train_otf.sh
#
# Quick smoke (1 case, short time, gh-dev partition):
#     see slurm_smoke_test.sh
# =============================================================================

#SBATCH --job-name=raft-otf
#SBATCH --output=logs/raft_otf_%A_%a.out
#SBATCH --error=logs/raft_otf_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # OTF workers need many CPU cores
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00             # safe margin over expected ~3-4h per case
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail

# ---- Paths -----------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$WORK/projects/RAFT-DVC}"
cd "$PROJECT_DIR"
mkdir -p logs

# ---- Activate conda env ----------------------------------------------------
# Source conda init explicitly so `conda activate` works inside non-interactive
# SBATCH shell (default /bin/bash is non-interactive, doesn't read .bashrc).
module load conda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/raft-dvc"

# ---- Parse case from case_matrix ------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Did you submit with --array?"
    exit 2
fi

eval "$(python scripts/phase1/case_helper.py \
    --matrix configs/phase1/case_matrix.yaml \
    --index "$SLURM_ARRAY_TASK_ID")"

# ---- Shared sweep params --------------------------------------------------
EPOCHS=500
NUM_WORKERS=32
MAX_LR="4.0e-4"             # sqrt scaling from baseline 2e-4 @ batch=8
OTF_TRAIN_LENGTH=1200       # 50 iters/epoch at batch=24 (drop_last clean)
OTF_VAL_LENGTH=96           # 4 batches at batch=24

# ---- Storage layout (per CLAUDE.md TACC policy) ---------------------------
# Training datasets are OTF -- NO disk reads needed.  Test sets only live on
# $SCRATCH (uploaded separately for the eventual OOD pass).
export DATA_PHASE1_ROOT="$SCRATCH/raft-dvc/data_phase1_test"
export CHECKPOINTS_ROOT="$WORK/checkpoints/raft-dvc"
mkdir -p "$CHECKPOINTS_ROOT/phase1"

# Reduce CUDA allocator fragmentation (defensive)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Banner ---------------------------------------------------------------
echo "===================================================================="
echo "Job ID         : $SLURM_JOB_ID"
echo "Array task     : $SLURM_ARRAY_TASK_ID  ($SLURM_ARRAY_JOB_ID)"
echo "Node           : $SLURM_NODELIST"
echo "Start time     : $(date)"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA driver    : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "Python         : $(which python)"
echo "PyTorch        : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "-------------------------------------------------------------------"
echo "Case           : $NAME"
echo "Data config    : $DATA"
echo "Model config   : $MODEL"
echo "Batch size     : $BATCH"
echo "Epochs         : $EPOCHS"
echo "Workers        : $NUM_WORKERS"
echo "max_lr         : $MAX_LR"
echo "OTF length T/V : $OTF_TRAIN_LENGTH / $OTF_VAL_LENGTH"
echo "CHECKPOINTS    : $CHECKPOINTS_ROOT/phase1/$NAME"
echo "===================================================================="

# ---- Run training ---------------------------------------------------------
python scripts/phase1/train_phase1.py \
    --model-config "$MODEL" \
    --data-config "$DATA" \
    --experiment-name "$NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --num-workers "$NUM_WORKERS" \
    --max-lr "$MAX_LR" \
    --otf \
    --otf-train-length "$OTF_TRAIN_LENGTH" \
    --otf-val-length "$OTF_VAL_LENGTH"

echo "Training done at $(date)"
echo "Final best ckpt: $CHECKPOINTS_ROOT/phase1/$NAME/best_model.pth"
