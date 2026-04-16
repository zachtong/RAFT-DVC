#!/bin/bash
# =============================================================================
# SLURM Job Script for RAFT-DVC Training on TACC Vista
# =============================================================================
# Usage:
#   sbatch scripts/slurm_train.sh
#
# Monitor:
#   squeue -u $USER                    # check job status
#   tail -f slurm_train_<jobid>.out    # watch output
#   scancel <jobid>                    # cancel job
# =============================================================================

#SBATCH --job-name=raft-dvc
#SBATCH --output=slurm_train_%j.out
#SBATCH --error=slurm_train_%j.err
#SBATCH --partition=gh                 # Vista GH200 GPU partition (gh-dev for short tests)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16             # for dataloader workers
#SBATCH --gres=gpu:1                   # 1 GPU (GH200 has 96GB HBM3)
#SBATCH --time=24:00:00               # max walltime (HH:MM:SS)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@utexas.edu   # change this

# ---- Environment setup ----
module load conda
conda activate $WORK/envs/raft-dvc

# ---- Paths ----
PROJECT_DIR=$WORK/projects/RAFT-DVC
DATA_DIR=$WORK/data/raft-dvc
SCRATCH_DIR=$SCRATCH/raft-dvc

# ---- Print job info ----
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

cd $PROJECT_DIR

# ---- Run training ----
# NOTE (2026-04-16): The legacy train_confocal.py was moved to
# archive/scripts_old/ as part of the Phase-1 reorganization. The Phase-1
# training entry point is scripts/phase1/train_phase1.py (to be written).
# Until that is in place, the two blocks below are commented out so sbatch
# won't silently run a no-op. Uncomment once train_phase1.py exists.

# python scripts/phase1/train_phase1.py \
#     --model-config configs/models/raft_dvc_1_8_p4_r4.yaml \
#     --training-config configs/phase1/phase1_r4_medium_size64_1_8.yaml

# Resume from checkpoint:
# python scripts/phase1/train_phase1.py \
#     --model-config configs/models/raft_dvc_1_8_p4_r4.yaml \
#     --training-config configs/phase1/phase1_r4_medium_size64_1_8.yaml \
#     --resume $SCRATCH_DIR/training_runs/<exp_name>/checkpoint_best.pth

echo "ERROR: slurm_train.sh is a template — edit it to call train_phase1.py before submitting."
exit 1

echo "Training finished at $(date)"
