#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- Ruler-law 4th point, Experiment D: 1/1 (f=1) @ input 32^3
# =============================================================================
# CLEAN comparison: f=1 on the EXACT SAME data as the 1/2 arm (r2_medium_size32,
# resolvable r2 beads, input disp [1,2] fm == [1,2] raw at f=1). feature 32^3 ->
# corr 32^6 = 4.3 GB -> gradient-checkpointed config. The 1/2 arm on this data
# gives raw-EPE 0.045 (feature-EPE 0.0225); the ruler law predicts f=1 -> ~0.022
# ~ classical ICGN 0.021, testing whether f=1 reaches classical precision on
# resolvable beads (no marginal-bead confound).
#
# Matched recipe to the arms: effective batch 8 (4 x accum 2), 300 epochs,
# max-lr 2e-4, pct-start 0.2, seed 42 -> total_steps = 300*(2000/4//2) = 75000.
#
# Usage:  sbatch scripts/phase1/slurm_1_1_r2at32_vista.sh
# Monitor: squeue -u $USER ; tail -f logs/rdvc_1_1_r2at32_<jobid>.out
# =============================================================================
#SBATCH --job-name=rdvc11r2
#SBATCH --output=logs/rdvc_1_1_r2at32_%j.out
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
EXP="paper1_1_1_r2at32"
EXP_DIR="$OUT_ROOT/phase1/$EXP"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT/phase1"
echo "=== Job $SLURM_JOB_ID on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
    --model-config configs/models/raft_dvc_1_1_p2_r4_ckpt.yaml \
    --data-config  r2_medium_size32 \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name "$EXP" \
    --epochs 300 --batch-size 4 --grad-accum-steps 2 \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed 42 $RESUME

echo "=== finished (or hit walltime) @ $(date) ==="
