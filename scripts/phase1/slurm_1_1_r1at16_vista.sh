#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- Ruler-law 4th point, Experiment A: 1/1 (f=1) @ input 16^3
# =============================================================================
# STRICT matched design: feature volume 16^3 (== the 32/64/128 arms), r1 beads,
# input disp [1,2]. corr = 16^6 = 67 MB -> trivial memory, no checkpointing.
# Tests whether the ruler law's 4th point at f=1 lands at raw-EPE ~ c ~ 0.017
# (caveat: r1 beads sit at the resolvability edge -- see preview).
#
# Matched training recipe to the arms: effective batch 8, 300 epochs, max-lr
# 2e-4, pct-start 0.2, seed 42 -> total_steps = 300*(2000/8) = 75000.
#
# Usage:  sbatch scripts/phase1/slurm_1_1_r1at16_vista.sh
# Monitor: squeue -u $USER ; tail -f logs/rdvc_1_1_r1at16_<jobid>.out
# =============================================================================
#SBATCH --job-name=rdvc11r1
#SBATCH --output=logs/rdvc_1_1_r1at16_%j.out
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
EXP="paper1_1_1_r1at16"
EXP_DIR="$OUT_ROOT/phase1/$EXP"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT/phase1"
echo "=== Job $SLURM_JOB_ID on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -d "$DATA_ROOT/r1_medium_size16/train" ]; then
    echo "[gen] building r1_medium_size16 on \$SCRATCH (compute-node CPUs)..."
    python scripts/paper1/gen_1_1_data.py --preset r1_16 --root "$DATA_ROOT"
fi
[ -d "$DATA_ROOT/r1_medium_size16/train" ] || { echo "ERROR: data gen failed at $DATA_ROOT/r1_medium_size16"; exit 1; }

if [ -f "$EXP_DIR/latest.pth" ]; then
    EP=$(python -c "import torch;print(int(torch.load('$EXP_DIR/latest.pth',map_location='cpu').get('epoch',0)))" 2>/dev/null || echo 0)
    if [ "${EP:-0}" -ge 299 ]; then echo "[done] epoch $EP -- complete"; exit 0; fi
    RESUME="--resume $EXP_DIR/latest.pth"; echo "[resume] epoch $EP"
else
    RESUME=""; echo "[fresh] starting from epoch 0"
fi

srun python -u scripts/phase1/train_phase1.py \
    --model-config configs/models/raft_dvc_1_1_p2_r4.yaml \
    --data-config  r1_medium_size16 \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name "$EXP" \
    --epochs 300 --batch-size 8 --grad-accum-steps 1 \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed 42 $RESUME

echo "=== finished (or hit walltime) @ $(date) ==="
