#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- Control C1 for the f=1 precision-floor study
# =============================================================================
# QUESTION. The f=1 arm sits 2.2x above the ladder even with the sensor noise
# switched off, and a classical-DVC radius sweep showed the data itself carries
# far more sub-voxel information than any arm extracts, so the excess belongs to
# the model. But f=1 differs from the ladder in TWO ways at once: its flow grid
# is not pooled, and it is the only arm whose convolutions run at full
# resolution. This run separates them.
#
# C1 = convolutions at FULL resolution, flow on the 1/2 grid.
#   Encoder is ShallowEncoder with the stride-2 moved from conv1 to layer3:
#   identical modules, identical 93,792 parameters, same output stride 2. The
#   only change is that the capacity is spent before pooling instead of after.
#
# READING.
#   C1 ~ the released s2 arm  -> convolution resolution is harmless, so the f=1
#                               excess is about the missing pooling, i.e. the
#                               physics the two-term floor law describes.
#   C1 markedly worse         -> the encoder style is the handicap, and no f=1
#                               arm in either paper can be used to test that law
#                               until this is controlled for.
#
# World is r2_medium_size32, the same data the released s2 arm trained on, so
# s2 is the baseline and needs no retraining.
#
# Recipe matched to the ladder: effective batch 8, 300 epochs, max-lr 2e-4,
# pct-start 0.2, seed 42 -> total_steps = 300*(2000/8) = 75000.
# Feature volume is 16^3, so the correlation is cheap and no checkpointing is
# needed; expect roughly the s2 arm's cost.
#
# Usage:  sbatch scripts/phase1/slurm_ctrl_c1_late_stride_vista.sh
# Monitor: squeue -u $USER ; tail -f logs/rdvc_ctrl_c1_<jobid>.out
# Pairs with: slurm_ctrl_c2_shallow_up_vista.sh (the other cell of the 2x2)
# =============================================================================
#SBATCH --job-name=rdvcC1
#SBATCH --output=logs/rdvc_ctrl_c1_%j.out
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/envs/raft-dvc"

PROJECT_DIR="$WORK/projects/RAFT-DVC"
DATA_ROOT="$SCRATCH/raft-dvc/data_paper1_axis2"
OUT_ROOT="$SCRATCH/raft-dvc/training_runs"
EXP="paper1_ctrl_c1_late_stride"
EXP_DIR="$OUT_ROOT/phase1/$EXP"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT/phase1"
echo "=== Job $SLURM_JOB_ID on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Same world as the released s2 arm.
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
    --model-config configs/models/raft_dvc_ctrl_late_stride.yaml \
    --data-config  r2_medium_size32 \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name "$EXP" \
    --epochs 300 --batch-size 8 --grad-accum-steps 1 \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed 42 $RESUME

echo "=== finished (or hit walltime) @ $(date) ==="
