#!/bin/bash
# =============================================================================
# TACC Vista GH200 -- multi-seed retraining of a ruler-law arm (statistical
# stability, reviewer point 3.6). Parameterized by ARM (s2/s4/s8) and SEED.
#
# All seeds of a given arm SHARE one self-generated dataset (fixed generation
# seed, idempotent skip-existing) so only the TRAINING randomness (init, batch
# order, augmentation) varies between seeds -- isolating between-seed variance
# from data-generation variance. Matched recipe: effective batch 8, 300 epochs,
# max-lr 2e-4, pct-start 0.2 -> total_steps 75000 (same as the original arms).
#
# Usage (launch a 3-seed sweep of all three arms = 9 jobs):
#   for A in s2 s4 s8; do for S in 42 43 44; do
#     sbatch --export=ALL,ARM=$A,SEED=$S scripts/phase1/slurm_arm_seed_vista.sh
#   done; done
# Monitor: squeue -u $USER ; tail -f logs/rdvc_seed_<ARM>_s<SEED>_<jobid>.out
# =============================================================================
#SBATCH --job-name=rdvcseed
#SBATCH --output=logs/rdvc_seed_%j.out
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zachtong@utexas.edu

set -euo pipefail
: "${ARM:?set ARM=s2|s4|s8 via --export}"
: "${SEED:?set SEED=<int> via --export}"

case "$ARM" in
  s2) CFG=raft_dvc_1_2_p2_r4.yaml; PRESET=r2_32;  DATACFG=r2_medium_size32;  BS=8; ACC=1 ;;
  s4) CFG=raft_dvc_1_4_p2_r4.yaml; PRESET=r4_64;  DATACFG=r4_medium_size64;  BS=8; ACC=1 ;;
  s8) CFG=raft_dvc_1_8_p2_r4.yaml; PRESET=r8_128; DATACFG=r8_medium_size128; BS=4; ACC=2 ;;
  *)  echo "ERROR: ARM must be s2|s4|s8 (got '$ARM')"; exit 1 ;;
esac

source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/envs/raft-dvc"
PROJECT_DIR="$WORK/projects/RAFT-DVC"
DATA_ROOT="$SCRATCH/raft-dvc/data_paper1_axis2"
OUT_ROOT="$SCRATCH/raft-dvc/training_runs"
EXP="paper1_${ARM}_seed${SEED}"
EXP_DIR="$OUT_ROOT/phase1/$EXP"
cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT/phase1"
echo "=== Job $SLURM_JOB_ID | ARM=$ARM SEED=$SEED | $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# One shared dataset per arm (fixed gen seed, idempotent). Concurrent seed jobs
# may race here; the skip-existing check tolerates a partially-built set, and a
# rerun completes it. If you prefer, pre-build once before launching the sweep.
if [ ! -d "$DATA_ROOT/$DATACFG/train" ] || \
   [ "$(ls "$DATA_ROOT/$DATACFG/train"/*.npz 2>/dev/null | wc -l)" -lt 2000 ]; then
    echo "[gen] building $DATACFG on \$SCRATCH ..."
    python scripts/paper1/gen_1_1_data.py --preset "$PRESET" --root "$DATA_ROOT"
fi

if [ -f "$EXP_DIR/latest.pth" ]; then
    EP=$(python -c "import torch;print(int(torch.load('$EXP_DIR/latest.pth',map_location='cpu').get('epoch',0)))" 2>/dev/null || echo 0)
    if [ "${EP:-0}" -ge 299 ]; then echo "[done] epoch $EP"; exit 0; fi
    RESUME="--resume $EXP_DIR/latest.pth"; echo "[resume] epoch $EP"
else
    RESUME=""; echo "[fresh]"
fi

srun python -u scripts/phase1/train_phase1.py \
    --model-config "configs/models/$CFG" \
    --data-config  "$DATACFG" \
    --data-root    "$DATA_ROOT" \
    --output-root  "$OUT_ROOT" \
    --experiment-name "$EXP" \
    --epochs 300 --batch-size "$BS" --grad-accum-steps "$ACC" \
    --max-lr 2.0e-4 --pct-start 0.2 \
    --num-workers 16 --latest-interval 1 --seed "$SEED" $RESUME

echo "=== finished (or hit walltime) @ $(date) ==="
