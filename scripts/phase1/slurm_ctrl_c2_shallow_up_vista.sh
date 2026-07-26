#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- Control C2 for the f=1 precision-floor study
# =============================================================================
# C2 = convolutions at 1/2 resolution, flow on the FULL grid.
#   Encoder is ShallowEncoder unchanged, including its initial stride-2 and its
#   93,792 parameters, with a trilinear x2 upsample appended so the feature map
#   and therefore the flow land on the full voxel grid. The convolutions see
#   pooled data exactly as the s2 arm's do; only the grid the flow is solved on
#   changes.
#
# Together with C1 and the two existing arms this closes a 2x2:
#
#                         convolutions at full res | convolutions at 1/2 res
#   flow grid f = 1   |   the f=1 arm              |   C2  (this run)
#   flow grid f = 2   |   C1                       |   the released s2 arm
#
# READING.
#   C2 ~ the released s2 arm -> solving the flow on the finer grid costs nothing
#                               by itself, so the two-term law's second term is
#                               far smaller in 3D than the f=1 arm suggests.
#   C2 degrades toward f=1   -> the penalty really is about the unpooled flow
#                               grid, which is the law's own mechanism.
#
# COST. Feature volume is 32^3, so the all-pairs correlation is 32^6 = 4.3 GB
# and the run is gradient-checkpointed, same class as the f=1 arm: about
# 14 min/epoch on GH200, so 300 epochs is roughly 71 h. That exceeds one
# walltime; the script resumes from latest.pth, so resubmit it until it prints
# "[done]". Chain it if your queue allows:
#   JID=$(sbatch --parsable scripts/phase1/slurm_ctrl_c2_shallow_up_vista.sh)
#   for i in 1 2 3; do JID=$(sbatch --parsable --dependency=afterany:$JID \
#       scripts/phase1/slurm_ctrl_c2_shallow_up_vista.sh); done
#
# Recipe matched to the ladder: effective batch 8, 300 epochs, max-lr 2e-4,
# pct-start 0.2, seed 42 -> total_steps = 300*(2000/8) = 75000.
#
# MEMORY. Measured peak for this config at input 32^3, iters 12, forward plus
# backward: 19.1 GB at micro-batch 1, 38.1 GB at 2, which extrapolates to about
# 76 GB at 4. That would fit GH200's 96 GB with only ~20% headroom, and the
# trainer's response to a CUDA OOM is to log a warning and DROP the whole
# accumulation group and carry on, so an occasional OOM would silently corrupt
# the very measurement this run exists to make. Micro-batch 2 with accumulation
# 4 keeps the effective batch and the optimizer-step count identical at a
# comfortable ~38 GB. If you would rather have the speed and are watching the
# log for "CUDA OOM", switch to --batch-size 4 --grad-accum-steps 2.
#
# Usage:  sbatch scripts/phase1/slurm_ctrl_c2_shallow_up_vista.sh
# Monitor: squeue -u $USER ; tail -f logs/rdvc_ctrl_c2_<jobid>.out
# =============================================================================
#SBATCH --job-name=rdvcC2
#SBATCH --output=logs/rdvc_ctrl_c2_%j.out
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/envs/raft-dvc"

PROJECT_DIR="$WORK/projects/RAFT-DVC"
DATA_ROOT="$SCRATCH/raft-dvc/data_paper1_axis2"
OUT_ROOT="$SCRATCH/raft-dvc/training_runs"
EXP="paper1_ctrl_c2_shallow_up"
EXP_DIR="$OUT_ROOT/phase1/$EXP"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT/phase1"
echo "=== Job $SLURM_JOB_ID on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Same world as C1 and as the released s2 arm.
if [ ! -d "$DATA_ROOT/r2_medium_size32/train" ]; then
    echo "[gen] building r2_medium_size32 on \$SCRATCH (compute-node CPUs)..."
    python scripts/paper1/gen_1_1_data.py --preset r2_32 --root "$DATA_ROOT"
fi
[ -d "$DATA_ROOT/r2_medium_size32/train" ] || { echo "ERROR: data gen failed at $DATA_ROOT/r2_medium_size32"; exit 1; }

if [ -f "$EXP_DIR/latest.pth" ]; then
    EP=$(python -c "import torch;print(int(torch.load('$EXP_DIR/latest.pth',map_location='cpu').get('epoch',0)))" 2>/dev/null || echo 0)
    if [ "${EP:-0}" -ge 299 ]; then echo "[done] epoch $EP -- complete"; exit 0; fi
    RESUME="--resume $EXP_DIR/latest.pth"; echo "[resume] epoch $EP"
else
    RESUME=""; echo "[fresh] starting from epoch 0"
fi

srun python -u scripts/phase1/train_phase1.py \
    --model-config configs/models/raft_dvc_ctrl_shallow_up.yaml \
    --data-config  r2_medium_size32 \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name "$EXP" \
    --epochs 300 --batch-size 2 --grad-accum-steps 4 \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed 42 $RESUME

echo "=== finished (or hit walltime) @ $(date) ==="
