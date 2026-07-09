#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- Paper-1 Axis-1 completing arm: 1/2 encoder @ input 64 (fm32)
# =============================================================================
# Continues the local RTX-5090 run (paper1_axis1_1_2_at64, seed 42, eff batch 8,
# pct_start 0.2) on GH200.  Vista has 96 GB HBM3, so the fm32 correlation fits
# WITHOUT gradient checkpointing at batch 2 -> plain config raft_dvc_1_2_p2_r4.yaml
# (not the *_ckpt variant), batch 2 x accum 4 = eff 8.  This keeps OneCycle
# total_steps identical to the local batch-1 x accum-8 run (2000/bs // accum *
# epochs = 250 * 300 = 75000 either way), so a checkpoint copied from the 5090
# resumes here with a valid scheduler state.
#
# 24 h walltime cap on `gh`; --latest-interval 1 + auto-resume make it safe to
# resubmit across windows (or chain with --dependency=afterany:<jobid>).
#
# Usage:  sbatch scripts/phase1/slurm_1_2_at64_vista.sh
# Monitor: squeue -u $USER ; tail -f logs/rdvc_1_2_at64_<jobid>.out
# =============================================================================
#SBATCH --job-name=rdvc12at64
#SBATCH --output=logs/rdvc_1_2_at64_%j.out
#SBATCH --error=logs/rdvc_1_2_at64_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/raft-dvc"

PROJECT_DIR="$WORK/projects/RAFT-DVC"
DATA_ROOT="$SCRATCH/raft-dvc/data_paper1_v3"
OUT_ROOT="$SCRATCH/raft-dvc/training_runs"
EXP="paper1_axis1_1_2_at64"
EXP_DIR="$OUT_ROOT/phase1/$EXP"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT/phase1"

echo "=== Job $SLURM_JOB_ID on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0))"
[ -d "$DATA_ROOT/r4_medium_size64/train" ] || { echo "ERROR: data missing at $DATA_ROOT/r4_medium_size64 -- see runbook step 3 (regenerate on \$SCRATCH)"; exit 1; }

# Auto-resume: from a checkpoint copied off the 5090 on the first submit, then
# from Vista's own latest.pth on every subsequent 24 h window.
RESUME=""
if [ -f "$EXP_DIR/latest.pth" ]; then
    RESUME="--resume $EXP_DIR/latest.pth"
    echo "[resume] $EXP_DIR/latest.pth"
else
    echo "[fresh]  no latest.pth found -- starting from epoch 0"
fi

srun python scripts/phase1/train_phase1.py \
    --model-config configs/models/raft_dvc_1_2_p2_r4.yaml \
    --data-config  r4_medium_size64 \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name "$EXP" \
    --epochs 300 --batch-size 2 --grad-accum-steps 4 \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed 42 $RESUME

echo "=== finished (or hit walltime) @ $(date) ==="
# If not yet at epoch 300, resubmit this same script -- it resumes from latest.pth.
