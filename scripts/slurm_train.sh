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
#SBATCH --partition=gpu-gh200          # Vista GPU partition (check with 'sinfo')
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
# Option 1: Use train_confocal.py with separate model + training configs
python scripts/train_confocal.py \
    --model-config configs/models/raft_dvc_1_8_p4_r4.yaml \
    --training-config configs/training/confocal_128_v1_1_8_p4_r4.yaml

# Option 2: Resume from checkpoint
# python scripts/train_confocal.py \
#     --model-config configs/models/raft_dvc_1_8_p4_r4.yaml \
#     --training-config configs/training/confocal_128_v1_1_8_p4_r4.yaml \
#     --resume $SCRATCH_DIR/training_runs/experiment/checkpoint_best.pth

echo "Training finished at $(date)"
