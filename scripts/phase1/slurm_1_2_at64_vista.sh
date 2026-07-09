#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- Paper-1 Axis-1 completing arm: 1/2 encoder @ input 64 (fm32)
# =============================================================================
# Continues the local RTX-5090 run (paper1_axis1_1_2_at64, seed 42, eff batch 8,
# pct_start 0.2) on GH200.  The GH200's GPU HBM is ~96 GB (nvidia-smi: 97871 MiB;
# the "120GB" in the device name is nominal).  Measured on an idev node 2026-07-09:
#   - batch 2 NON-checkpointed  -> OOM (the 12-iter grid_sample backward buffers
#     exceed 96 GB); batch 1 non-ckpt fits (~65 GB, 3.08 samp/s).
#   - CHECKPOINTED batch 4 x accum 2 fits (~66 GB) AND is faster (4.0 samp/s,
#     ~8.3 min/epoch vs ~10.8) -> use it.
# Same config as the local run (raft_dvc_1_2_p2_r4_ckpt.yaml), just batch 4 vs 1,
# so weights/optimizer/scheduler transfer identically.  batch 4 x accum 2 keeps
# OneCycle total_steps = 300 * (2000/bs // accum) = 300 * 250 = 75000, identical
# to the local batch 1 x accum 8 run, so the copied checkpoint resumes on the
# exact same LR curve.  (The trainer's OOM-immunity is a backstop only; on a
# clean exclusive SLURM GPU there should be zero drops.)
#
# 24 h walltime cap on `gh`; --latest-interval 1 + auto-resume make it safe to
# resubmit across windows (or chain with --dependency=afterany:<jobid>).
#
# Usage:  sbatch scripts/phase1/slurm_1_2_at64_vista.sh
# Monitor: squeue -u $USER ; tail -f logs/rdvc_1_2_at64_<jobid>.out
# =============================================================================
#SBATCH --job-name=rdvc12at64
#SBATCH --output=logs/rdvc_1_2_at64_%j.out
# NOTE: no --error line on purpose -- SLURM then merges stderr (where Python's
# logging writes the per-epoch INFO lines) into the .out file, so `tail -f .out`
# shows the training progress, not just the startup banner.
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# NOTE: no --gres on Vista; the `gh` partition allocates a whole GH200 node
# (GPU included).  --gres=gpu:1 triggers "Invalid generic resource (gres)".
#SBATCH --time=24:00:00
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
EXP="paper1_axis1_1_2_at64"
EXP_DIR="$OUT_ROOT/phase1/$EXP"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT/phase1"

echo "=== Job $SLURM_JOB_ID on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0))"
[ -d "$DATA_ROOT/r4_medium_size64/train" ] || { echo "ERROR: data missing at $DATA_ROOT/r4_medium_size64 -- see runbook step 3 (regenerate on \$SCRATCH)"; exit 1; }

# Completion guard: if latest.pth already reached the final epoch, this job (and
# any trailing chained jobs) has nothing to do -- exit fast instead of spinning
# up training to run zero epochs. Fails open (EP=0 -> runs) if the load errors.
if [ -f "$EXP_DIR/latest.pth" ]; then
    EP=$(python -c "import torch;print(int(torch.load('$EXP_DIR/latest.pth',map_location='cpu').get('epoch',0)))" 2>/dev/null || echo 0)
    if [ "${EP:-0}" -ge 299 ]; then
        echo "[done] latest.pth at epoch $EP (>=299) -- 300-epoch run complete, exiting."
        exit 0
    fi
    echo "[guard] latest.pth at epoch $EP -- will resume."
fi

# Auto-resume: from a checkpoint copied off the 5090 on the first submit, then
# from Vista's own latest.pth on every subsequent 24 h window.
RESUME=""
if [ -f "$EXP_DIR/latest.pth" ]; then
    RESUME="--resume $EXP_DIR/latest.pth"
    echo "[resume] $EXP_DIR/latest.pth"
else
    echo "[fresh]  no latest.pth found -- starting from epoch 0"
fi

srun python -u scripts/phase1/train_phase1.py \
    --model-config configs/models/raft_dvc_1_2_p2_r4_ckpt.yaml \
    --data-config  r4_medium_size64 \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name "$EXP" \
    --epochs 300 --batch-size 4 --grad-accum-steps 2 \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed 42 $RESUME

echo "=== finished (or hit walltime) @ $(date) ==="
# If not yet at epoch 300, resubmit this same script -- it resumes from latest.pth.
