#!/bin/bash
# =============================================================================
# TACC Vista -- VolRAFT architecture, sampler bug FIXED, trained on OUR
# r8_medium_size128 bead data (identical data+budget to our 1/8 arm).
# Primary retrained baseline for the paper-1 benchmark.
# Usage:  sbatch --dependency=afterok:<GEN_JOBID> scripts/paper1/slurm_volraft_fixed_vista.sh
# =============================================================================
#SBATCH --job-name=vrf-fixed
#SBATCH --output=logs/volraft_fixed_%j.out
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
DATA_DIR="$SCRATCH/raft-dvc/data_paper1_axis2/r8_medium_size128"
OUT_DIR="$SCRATCH/raft-dvc/training_runs/phase1/volraft_fixed_r8md128"
cd "$PROJECT_DIR"
mkdir -p logs "$OUT_DIR"

echo "=== Job $SLURM_JOB_ID on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -d "$DATA_DIR/train" ] || { echo "ERROR: data missing at $DATA_DIR"; exit 1; }

RESUME=""
if [ -f "$OUT_DIR/latest.pth" ]; then
    EP=$(python -c "import torch;print(int(torch.load('$OUT_DIR/latest.pth',map_location='cpu').get('epoch',0)))" 2>/dev/null || echo 0)
    if [ "${EP:-0}" -ge 299 ]; then echo "[done] epoch $EP -- complete."; exit 0; fi
    RESUME="--resume $OUT_DIR/latest.pth"
    echo "[resume] epoch $EP"
fi

srun python -u volraft_fixed/train_volraft_fixed.py \
    --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" \
    --epochs 300 --batch-size 8 --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --seed 42 $RESUME

echo "=== finished (or walltime) @ $(date) ==="
