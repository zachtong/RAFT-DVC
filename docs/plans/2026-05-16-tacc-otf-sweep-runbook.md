# TACC Vista GH200 OTF Sweep Runbook

**Date:** 2026-05-16
**Goal:** Train 20 Phase-1 cases (size16 + size32, OTF, 500 epoch) on TACC GH200.
**Email:** zachtong@utexas.edu

---

## 0. Background facts (recap)

| Resource | Value |
|---|---|
| Login host | `vista.tacc.utexas.edu` |
| GPU | NVIDIA H100 80GB Hopper (sm_90) per GH200 superchip |
| GPU mem | 96 GB HBM3 |
| CPU | Grace 72-core ARM Neoverse-V2 |
| CPU mem | 480 GB DDR5 |
| Filesystems | `$HOME` (10 GB), `$WORK` (1 TB), `$SCRATCH` (multi-TB, 10-day purge) |
| Partitions | `gh` (24h max), `gh-dev` (30 min, for smoke tests) |
| Auth | TACC username + password + RSA token (MFA) |

---

## 1. Login

```bash
ssh -X zachtong@vista.tacc.utexas.edu
# Will prompt: password, then RSA token
```

Quick sanity:

```bash
echo "HOME=$HOME"
echo "WORK=$WORK"
echo "SCRATCH=$SCRATCH"
whoami           # should show: zachtong (or your TACC username)
sinfo -s         # cluster status
```

---

## 2. First-time setup (only if env doesn't exist)

Verify env presence:

```bash
ls -la $WORK/envs/raft-dvc
ls -la $WORK/projects/RAFT-DVC
```

### 2a. If `$WORK/envs/raft-dvc` does not exist

```bash
module load conda
conda create -p $WORK/envs/raft-dvc python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $WORK/envs/raft-dvc

# PyTorch for GH200 (sm_90, Hopper) -- stable cu124 wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify GPU access (may need an interactive node; see Step 5)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2b. Project code

```bash
cd $WORK/projects
# If you haven't cloned yet:
git clone <YOUR_REPO_URL> RAFT-DVC
cd RAFT-DVC

# Install Python deps (PyTorch already installed in 2a)
conda activate $WORK/envs/raft-dvc
pip install -r requirements.txt
```

**Important about numpy/scipy compatibility** (we hit this on the 5090):
`requirements.txt` pins `numpy<2.0`.  Stick with that — older scipy is
fine; numpy 2.x breaks scipy < 1.13.

### 2c. Verify imports

```bash
cd $WORK/projects/RAFT-DVC
python -c "
import torch, scipy, numpy, matplotlib, yaml, tensorboard
print('torch', torch.__version__)
print('scipy', scipy.__version__)
print('numpy', numpy.__version__)
"
# Plus our project modules:
python -c "
import sys; sys.path.insert(0, '.')
from src.data import safe_phase1_collate, build_otf_dataset
from src.visualization import render_flow_eval_4x3
print('Project imports OK')
"
```

---

## 3. Pull latest code (every push from local)

```bash
cd $WORK/projects/RAFT-DVC
git fetch origin
git status                # confirm clean
git pull origin master    # or your active branch
```

If you have local uncommitted changes on TACC (you shouldn't, but...):

```bash
git stash                 # save
git pull
git stash pop             # restore
```

---

## 4. Update conda env (after pulling new deps)

Only needed if `requirements.txt` changed:

```bash
conda activate $WORK/envs/raft-dvc
pip install -r requirements.txt --upgrade-strategy only-if-needed
```

---

## 5. Smoke test (gh-dev, 30 min)

**Always run this before submitting the full sweep**, especially after any
code change.  Catches OTF / env / GPU issues without queuing 20 jobs.

```bash
cd $WORK/projects/RAFT-DVC
sbatch scripts/phase1/slurm_smoke_test.sh
squeue -u $USER          # see job state (PD=pending, R=running, CG=completing)
```

When it starts (state goes from PD → R):

```bash
# Watch progress live
tail -f logs/raft_smoke_<jobid>.out
```

### Smoke test pass criteria

- [ ] Banner prints `CUDA available : True`
- [ ] First training line `Epoch 0 [0/...] Loss: ...` appears within 2 min
- [ ] Per-epoch time is **20-60 seconds** (GH200 should be faster than 5090's 36s)
- [ ] No `OutOfMemoryError`
- [ ] No `nan` in any loss
- [ ] `Validation - Loss: ... EPE: X` at epoch 4 with `X < 1.5`
- [ ] Final log line: `Smoke test done at ...`

### If smoke test fails

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA available : False` | wrong torch wheel / driver | Reinstall torch with cu124 |
| `ModuleNotFoundError: scipy` | env broken | redo Step 2a or `pip install -r requirements.txt` |
| `OutOfMemoryError` | batch too large for GH200 (unlikely) | drop to `--batch-size 16` |
| First epoch takes 5+ min | CPU starvation / num_workers wrong | reduce `--num-workers 16` |
| `Ninja is required` (only if rebuilding CUDA ext) | not relevant for OTF, ignore |  |
| Job stays in PD for > 30 min | queue backlog or wrong allocation | `scontrol show job <id>` to debug |

---

## 6. Submit the full 20-case sweep

Once smoke test passes:

```bash
cd $WORK/projects/RAFT-DVC
bash scripts/phase1/submit_sweep.sh
# Interactive prompt confirms; press 'y'
```

This calls `sbatch --array=0-19%4 scripts/phase1/slurm_train_otf.sh`.

### Monitoring

```bash
# Whole queue view (your jobs only)
squeue -u $USER

# Watch one task's live output
tail -f logs/raft_otf_<arrayid>_<taskid>.out

# All tasks status
squeue -u $USER --format='%.18i %.20j %.8u %.2t %.10M %.6D %R'

# Cancel one task or whole array
scancel <jobid>_<taskid>         # one task
scancel <jobid>                  # whole array
```

### Resume only failed tasks

If e.g. tasks 3, 7, 12 failed:

```bash
sbatch --array=3,7,12%4 scripts/phase1/slurm_train_otf.sh
```

---

## 7. Storage layout (where things go)

```
$WORK/projects/RAFT-DVC/              <- code (git, persistent)
$WORK/envs/raft-dvc/                  <- conda env (persistent)
$WORK/checkpoints/raft-dvc/phase1/    <- TRAINED CKPTS (small, persistent)
  ├── r2sp16_e11_otf/
  │   ├── best_model.pth              <- main deliverable, ~25 MB
  │   ├── latest.pth
  │   ├── checkpoint_epoch_NNNN.pth   <- every 50 epoch, ~25 MB each
  │   ├── training.log
  │   └── logs/                        <- TensorBoard events
  ├── r2md16_e11_otf/
  └── ...

$SCRATCH/raft-dvc/data_phase1_test/   <- NPZ test sets only (for later OOD)
                                         not needed for training (OTF)

$PROJECT_DIR/logs/                    <- SLURM stdout/stderr
  ├── raft_otf_<jobid>_0.out
  ├── raft_otf_<jobid>_1.out
  └── ...
```

### Disk budget after 20 cases

- ckpts: 20 cases × ~250 MB (10 ckpts × 25 MB) ≈ **5 GB on $WORK**
- TB events: 20 × ~50 MB ≈ 1 GB
- SLURM logs: negligible

Total $WORK growth: ~6 GB.  $WORK has 1 TB — no concern.

---

## 8. Download results back to local

After sweep finishes (or once each case finishes), pull ckpts back:

```bash
# From your local machine (in another terminal)
rsync -avz \
    zachtong@vista.tacc.utexas.edu:'$WORK/checkpoints/raft-dvc/phase1/' \
    "C:/Users/zt3323/OneDrive - The University of Texas at Austin/Documents/Python Codes/RAFT-DVC/checkpoints/phase1/"
```

(Use git-bash or WSL for rsync on Windows; or `scp -r` as fallback.)

You probably only need the `best_model.pth` of each case for the final OOD pass:

```bash
rsync -avz --include='*/' --include='best_model.pth' --include='training.log' --exclude='*' \
    zachtong@vista.tacc.utexas.edu:'$WORK/checkpoints/raft-dvc/phase1/' \
    ./local_phase1_ckpts/
```

---

## 9. TensorBoard

Two options:

### Option A: download events first, then view locally

```bash
rsync -avz \
    zachtong@vista.tacc.utexas.edu:'$WORK/checkpoints/raft-dvc/phase1/*/logs/' \
    ./tacc_tb_events/
tensorboard --logdir tacc_tb_events --port 6006
```

### Option B: SSH tunnel to TACC login node

```bash
# On local machine
ssh -L 6006:localhost:6006 zachtong@vista.tacc.utexas.edu
# Now on TACC:
module load conda
conda activate $WORK/envs/raft-dvc
cd $WORK/checkpoints/raft-dvc/phase1
tensorboard --logdir . --port 6006 --bind_all
# Open http://localhost:6006 in your local browser
```

> Warning: don't run TensorBoard on a compute node (queue resources).
> Run on login node only.

---

## 10. Common pitfalls & fixes

### conda activate fails in batch script
- **Symptom:** `CommandNotFoundError: Your shell has not been properly configured`
- **Fix:** the SLURM scripts already do `source "$(conda info --base)/etc/profile.d/conda.sh"` before `conda activate`. If you see this error, the conda module didn't load — make sure `module load conda` runs first.

### Job stays in PD for hours
- Probably TACC queue backlog. Check `sinfo -p gh` for partition state.
- Or your fair-share priority is low — check `sshare -u zachtong`.
- Reduce concurrency: resubmit with `CONCURRENT=2 bash submit_sweep.sh`.

### Email notifications not arriving
- Verify mail-user in `slurm_train_otf.sh` and `slurm_smoke_test.sh`.
- TACC mail may be flagged as spam — check `zachtong@utexas.edu` spam folder.

### Code mismatch (TACC vs local)
- TACC pulls from git, your local is OneDrive-synced.
- Always: edit locally → commit → push → `git pull` on TACC.
- Don't edit on TACC directly (no OneDrive sync there).

### `OneDrive` paths in `_base.yaml` baked into ckpts
- Not an issue — `train_phase1.py` resolves output paths at runtime via env vars
  (`CHECKPOINTS_ROOT`, `DATA_PHASE1_ROOT`) which the SLURM script sets correctly.

### `$WORK` quota near full
- Run `quota -s` to check.
- Delete old ckpts: `rm -rf $WORK/checkpoints/raft-dvc/phase1/old_run/`

### Walltime exceeded (8h cap in our script)
- A 500-epoch case taking > 8h means something is wrong with GH200 perf
  (or num_workers / OTF bottleneck).
- Inspect `tail -n 100 logs/raft_otf_<id>_<task>.out` for per-epoch times.

### NPZ test data not on $SCRATCH
- Doesn't matter for **training** (OTF doesn't read NPZ).
- Only matters when running OOD evaluation later — at which point you need
  to `rsync -avz local/data_phase1/*/test/ tacc:$SCRATCH/raft-dvc/data_phase1_test/`.

### LR / loss is NaN
- `--no-amp` to disable mixed precision (slower but stable).
- Or lower `--max-lr` to 2.8e-4 (less aggressive sqrt scaling).

### `Address already in use` for TensorBoard
- Another tensorboard is running. `pkill -f tensorboard` or pick a different port.

---

## 11. Files in this run

| File | Purpose |
|---|---|
| `configs/phase1/case_matrix.yaml` | 20 cases definition |
| `scripts/phase1/case_helper.py` | read case by index |
| `scripts/phase1/slurm_smoke_test.sh` | 1-case test on gh-dev |
| `scripts/phase1/slurm_train_otf.sh` | array job for full sweep |
| `scripts/phase1/submit_sweep.sh` | convenience submit wrapper |
| `scripts/phase1/train_phase1.py` | (existing) with `--max-lr` CLI added |
| `src/data/otf_dataset.py` | (existing) OTF dataset (fixed call_count bug) |
| `src/data/collate.py` | (existing) shared safe collate |
| `src/visualization/eval_panels.py` | (existing, used by OOD eval) |

---

## 12. Quick command reference

```bash
# SETUP (once)
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $WORK/envs/raft-dvc

# PULL CODE
cd $WORK/projects/RAFT-DVC && git pull

# SMOKE TEST
sbatch scripts/phase1/slurm_smoke_test.sh
squeue -u $USER

# FULL SWEEP
bash scripts/phase1/submit_sweep.sh

# MONITOR
squeue -u $USER
tail -f logs/raft_otf_<jobid>_0.out

# CANCEL
scancel <jobid>          # whole array
scancel <jobid>_3        # one task

# CLEANUP FAILED RUNS
rm -rf $WORK/checkpoints/raft-dvc/phase1/<failed_case_name>
```
