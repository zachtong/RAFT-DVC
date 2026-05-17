#!/bin/bash
# =============================================================================
# Paper-2 displacement scan: 20 cases (1/2 encoder x 4 configs x 5 disp levels)
# =============================================================================
# Purpose: probe the learnability viability boundary predicted by the
# paper-2 core thesis (fm-richness x fm-disp-resolvability formula).
# See notes/TACC_workflow_runbook.md and memory/paper2_core_thesis.md.
#
# Each task trains one (data_config, disp_range) pair on 1/2 encoder.
# Reads from configs/phase1/case_matrix_disp_scan.yaml.
#
# Submit with:
#     sbatch --array=0-19%4 scripts/phase1/slurm_disp_scan.sh
# Or use the wrapper:
#     bash scripts/phase1/submit_disp_scan.sh
#
# Differs from slurm_train_otf.sh:
# - Uses case_matrix_disp_scan.yaml (5 extra fields per case)
# - Passes --max-lr and OTF lengths derived from case
# - input-disp range is per-case, so we use input-disp mode for OTF
#   (requires OTF dataset support for input_disp_range -- see below).
#
# IMPORTANT: OTF dataset currently uses default fm-disp mode in
# Phase1SampleGenerator.  To use input-disp mode, set
# `feature_map_size = input_size / encoder_factor` and pass
# input_disp_min/max via env vars to a small wrapper.  Since this is
# complex, we instead pre-generate NPZ data for each (config, disp)
# combination -- simpler and reproducible.
# =============================================================================

#SBATCH --job-name=raft-ds
#SBATCH --output=logs/raft_ds_%A_%a.out
#SBATCH --error=logs/raft_ds_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu
# Note: gh partition implies 1 GH200 GPU per node -- do NOT add --gres.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$WORK/projects/RAFT-DVC}"
cd "$PROJECT_DIR"
mkdir -p logs

# ---- Activate conda env (Vista has no `conda` module) ---------------------
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

# ---- Parse case -----------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Did you submit with --array?"
    exit 2
fi

eval "$(python scripts/phase1/case_helper.py \
    --matrix configs/phase1/case_matrix_disp_scan.yaml \
    --index "$SLURM_ARRAY_TASK_ID")"

# Required exports from case (set by eval above):
#   NAME, DATA, MODEL, BATCH, MAX_LR, INPUT_DISP_MIN, INPUT_DISP_MAX

# ---- Shared sweep params --------------------------------------------------
EPOCHS=500
NUM_WORKERS=32
OTF_TRAIN_LENGTH=1200
OTF_VAL_LENGTH=96

# Paper-2 data lives separately (different disp ranges per case).
# Datasets must be pre-generated for each (config, disp) by the user
# OR we use OTF training -- but OTF currently uses default fm-disp
# range.  Easiest: use NPZ data pre-generated on TACC SCRATCH.
#
# Convention: paper-2 NPZ data lives at
#   $SCRATCH/raft-dvc/data_paper2_disp_scan/<DATA>_size64_d<MAX>/{train,val}
# Where <DATA> is e.g. "r4_medium" and <MAX> is input_disp_max (rounded int).
#
# For OTF use here (faster, no pre-gen), we directly feed input-disp
# range to a future OTF wrapper.  Until that wrapper exists, this script
# expects pre-generated NPZ.  See notes for pre-generation commands.

DISP_TAG="d$(printf '%g' "$INPUT_DISP_MAX")"
DATA_CONFIG_FULL="${DATA}_size64_${DISP_TAG}"

export DATA_PHASE1_ROOT="$SCRATCH/raft-dvc/data_paper2_disp_scan"
export CHECKPOINTS_ROOT="$WORK/checkpoints/raft-dvc"
mkdir -p "$CHECKPOINTS_ROOT/phase1"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Banner ---------------------------------------------------------------
echo "===================================================================="
echo "Job ID         : $SLURM_JOB_ID"
echo "Array task     : $SLURM_ARRAY_TASK_ID  ($SLURM_ARRAY_JOB_ID)"
echo "Node           : $SLURM_NODELIST"
echo "Start time     : $(date)"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "PyTorch        : $(python -c 'import torch; print(torch.__version__)')"
echo "-------------------------------------------------------------------"
echo "Case           : $NAME"
echo "Data raw cfg   : $DATA"
echo "Disp range     : [${INPUT_DISP_MIN}, ${INPUT_DISP_MAX}] voxels (input)"
echo "Data dir       : $DATA_PHASE1_ROOT/$DATA_CONFIG_FULL"
echo "Model          : $MODEL"
echo "Batch / LR     : $BATCH / $MAX_LR"
echo "Epochs / Workers: $EPOCHS / $NUM_WORKERS"
echo "CHECKPOINTS    : $CHECKPOINTS_ROOT/phase1/$NAME"
echo "===================================================================="

# Verify data exists (NPZ pre-generated)
if [ ! -d "$DATA_PHASE1_ROOT/$DATA_CONFIG_FULL/train" ]; then
    echo "ERROR: training data not found at $DATA_PHASE1_ROOT/$DATA_CONFIG_FULL/train"
    echo "Generate it first with:"
    echo "  python scripts/generate_phase1_dataset.py \\"
    echo "    --sizes 64 --train 1200 --val 100 --test 100 \\"
    echo "    --only-config $DATA --input-disp-range $INPUT_DISP_MIN $INPUT_DISP_MAX \\"
    echo "    --no-mixed --no-viz --output-root $DATA_PHASE1_ROOT/_tmp_$NAME"
    echo "  mv $DATA_PHASE1_ROOT/_tmp_$NAME/${DATA}_size64 $DATA_PHASE1_ROOT/$DATA_CONFIG_FULL"
    exit 3
fi

# ---- Train ----------------------------------------------------------------
python scripts/phase1/train_phase1.py \
    --model-config "$MODEL" \
    --data-config "$DATA_CONFIG_FULL" \
    --experiment-name "$NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --num-workers "$NUM_WORKERS" \
    --max-lr "$MAX_LR"

echo "Training done at $(date)"
echo "Best ckpt: $CHECKPOINTS_ROOT/phase1/$NAME/best_model.pth"
