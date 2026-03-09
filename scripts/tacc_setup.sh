#!/bin/bash
# =============================================================================
# TACC Vista HPC Setup Cheatsheet for RAFT-DVC
# =============================================================================
# Copy-paste each block below one at a time on Vista.
# Lines starting with '#' are comments — no need to run them.
#
# TACC storage quick reference:
#   $HOME    (~10 GB, no purge)   -> configs, .bashrc
#   $WORK    (~1 TB, no purge)    -> code, envs, permanent data
#   $SCRATCH (large, ~10d purge)  -> training runs, generated data
# =============================================================================


# ---- Step 1: Verify TACC environment ----
echo "HOME=$HOME  WORK=$WORK  SCRATCH=$SCRATCH"


# ---- Step 2: Create $WORK directories (persistent) ----
mkdir -p $WORK/projects
mkdir -p $WORK/envs
mkdir -p $WORK/data/raft-dvc/confocal
mkdir -p $WORK/data/raft-dvc/experimental
mkdir -p $WORK/checkpoints/raft-dvc
mkdir -p $WORK/shared_libs


# ---- Step 3: Create $SCRATCH directories (temporary) ----
mkdir -p $SCRATCH/raft-dvc/generated_data
mkdir -p $SCRATCH/raft-dvc/training_runs
mkdir -p $SCRATCH/raft-dvc/inference_output


# ---- Step 4: Clone project ----
cd $WORK/projects
git clone <YOUR_REPO_URL> RAFT-DVC


# ---- Step 5: Symlink scratch into project ----
ln -s $SCRATCH/raft-dvc $WORK/projects/RAFT-DVC/scratch
# Verify:
ls -la $WORK/projects/RAFT-DVC/scratch


# ---- Step 6: Create conda environment ----
module load conda                              # or: module load anaconda3
conda create -p $WORK/envs/raft-dvc python=3.10


# ---- Step 7: Activate env and install dependencies ----
conda activate $WORK/envs/raft-dvc

# GH200 (sm_90) works with stable PyTorch — no nightly needed
# Run 'nvidia-smi' to check CUDA driver version, then pick cu121 or cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
cd $WORK/projects/RAFT-DVC
pip install -r requirements.txt


# ---- Step 8: Verify GPU access ----
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"


# ---- Step 9: Add aliases to .bashrc ----
cat >> ~/.bashrc << 'EOF'

# --- RAFT-DVC TACC aliases ---
export PROJECTS=$WORK/projects
export DATA=$WORK/data
export CHECKPOINTS=$WORK/checkpoints
alias cdraft="cd $WORK/projects/RAFT-DVC"
alias actraft="conda activate $WORK/envs/raft-dvc"
alias cddata="cd $WORK/data"
alias cdscratch="cd $SCRATCH/raft-dvc"
# --- end RAFT-DVC aliases ---
EOF

source ~/.bashrc


# ---- Done! Quick reference ----
# cdraft    -> cd to project
# actraft   -> activate raft-dvc conda env
# cddata    -> cd to $WORK/data
# cdscratch -> cd to $SCRATCH/raft-dvc
#
# Directory layout:
#
# $WORK/
# ├── projects/RAFT-DVC/       <- code (git repo)
# │   └── scratch -> $SCRATCH/raft-dvc/
# ├── envs/raft-dvc/           <- conda env
# ├── data/
# │   ├── raft-dvc/
# │   │   ├── confocal/        <- permanent datasets
# │   │   └── experimental/
# │   └── future-project/      <- other projects' data
# ├── checkpoints/raft-dvc/    <- trained model weights
# └── shared_libs/
#
# $SCRATCH/raft-dvc/
# ├── generated_data/          <- synthetic data output
# ├── training_runs/           <- active training logs
# └── inference_output/        <- inference results
