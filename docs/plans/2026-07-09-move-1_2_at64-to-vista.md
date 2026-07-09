# Move the 1/2 @ input-64 (fm32) arm from the RTX-5090 to TACC Vista

**Date:** 2026-07-09
**Goal:** Finish `paper1_axis1_1_2_at64` (the fm32 arm that completes the Axis-1
3-point curve) on Vista GH200, **resuming** the local epoch-~52 checkpoint, so
the 5090 can be freed for other ML work. Confirm the Vista run is healthy
*before* stopping the local one.

**Config choice (measured on an idev node 2026-07-09):** the GH200's GPU HBM is
**~96 GB** (nvidia-smi: 97871 MiB; the "120GB" in the device name is nominal).
- batch 2 NON-checkpointed OOMs (the 12-iter grid_sample backward buffers exceed
  96 GB); batch 1 non-ckpt fits but underutilises the GPU (3.08 samp/s).
- **CHECKPOINTED batch 4 x accum 2 fits (~66 GB) and is faster (4.0 samp/s,
  ~8.3 min/epoch vs ~10.8)** -> this is the chosen config.

This is the **same** `raft_dvc_1_2_p2_r4_ckpt.yaml` as the local run, just batch 4
instead of batch 1, so weights/optimizer/scheduler transfer identically. Gradient
checkpointing only changes the backward compute/memory tradeoff, not the math.
batch 4 x accum 2 keeps OneCycle `total_steps` = 300 * (2000/bs // accum) =
300 * 250 = **75000**, identical to the local batch 1 x accum 8 run, so the copied
checkpoint's scheduler resumes on the exact same LR curve.

---

## Facts (from the 2026-05-16 runbook, still current)

| Resource | Value |
|---|---|
| Login | `ssh zachtong@vista.tacc.utexas.edu` (password + RSA token) |
| GPU | H100 80 GB Hopper (sm_90), **96 GB HBM3** per GH200 |
| Env | `$WORK/envs/raft-dvc` -- **stable** torch cu124 (NO nightly, unlike 5090) |
| Code | `$WORK/projects/RAFT-DVC` (git pull from `zachtong/RAFT-DVC`) |
| Data | `$SCRATCH/raft-dvc/data_paper1_v3` (regenerated, deterministic) |
| Runs | `$SCRATCH/raft-dvc/training_runs/phase1/paper1_axis1_1_2_at64` |
| Partitions | `gh` (24 h cap, real run), `gh-dev` (30 min, smoke) |

---

## Step 0 -- (local, DONE by Claude) push the training code

The two commits with `--pct-start`, `--latest-interval`, grad-accum, the
checkpoint-key fix, and OOM hardening are pushed to `origin/master`, plus the
Vista SLURM scripts + this runbook. Nothing further needed locally in this step.

## Step 1 -- Vista: pull code + verify env

```bash
ssh zachtong@vista.tacc.utexas.edu
cd $WORK/projects/RAFT-DVC
git fetch origin && git pull origin master
git log --oneline -3          # expect the two training commits + the SLURM commit

# Vista has NO `conda` module -- conda is auto-activated as (base) via .bashrc.
# Just activate the env directly (base is already on PATH):
conda activate $WORK/envs/raft-dvc
python -c "import torch,scipy,numpy,yaml;print('torch',torch.__version__,'numpy',numpy.__version__)"
python -c "import sys;sys.path.insert(0,'.');from src.core.raft_dvc import RAFTDVC;print('RAFTDVC import OK')"
```

If `$WORK/envs/raft-dvc` is gone (purged/never made), rebuild it -- see
`scripts/tacc_setup.sh` (§2a): conda create py3.10 + `pip install torch
torchvision --index-url https://download.pytorch.org/whl/cu124` +
`pip install -r requirements.txt` (numpy<2.0 is pinned; keep it).

## Step 2 -- Vista: regenerate the training data on $SCRATCH

Deterministic generator -> byte-identical to the local `data_paper1_v3`
(seeded by config hash). ~10-15 min on Grace's 72 cores.

```bash
cd $WORK/projects/RAFT-DVC
python scripts/generate_phase1_dataset.py \
    --only-config r4_medium --sizes 64 --input-disp-range 4 8 \
    --train 2000 --val 200 --test 0 \
    --output-root $SCRATCH/raft-dvc/data_paper1_v3 --no-mixed --no-viz
ls $SCRATCH/raft-dvc/data_paper1_v3/r4_medium_size64/{train,val} | head
```

## Step 3 -- copy the local epoch-~52 checkpoint to Vista (enables resume)

The local run saves `latest.pth` every epoch. Copy it (and `best_model.pth`)
into the Vista run dir so the SLURM script auto-resumes from it.

**From a local git-bash / WSL terminal** (rsync; scp works too):

```bash
SRC="/c/Zixiang_local_data/raft-dvc/paper1/phase1/paper1_axis1_1_2_at64"
DST='zachtong@vista.tacc.utexas.edu:$SCRATCH/raft-dvc/training_runs/phase1/paper1_axis1_1_2_at64'
ssh zachtong@vista.tacc.utexas.edu 'mkdir -p $SCRATCH/raft-dvc/training_runs/phase1/paper1_axis1_1_2_at64'
rsync -avz "$SRC/latest.pth" "$SRC/best_model.pth" "$DST/"
```

> The 5090 keeps training meanwhile -- that's fine. Whatever epoch `latest.pth`
> is at when you copy it is where Vista picks up. You can re-copy a fresher
> `latest.pth` right before the real submit to hand off more progress.

## Step 4 -- Vista: smoke test (OPTIONAL -- already verified on idev 2026-07-09)

The chosen config (ckpt batch 4 x accum 2) was already run on an idev GH200 node:
~66 GB, 4.0 samp/s, no OOM. So this sbatch smoke is optional insurance. If you
want it anyway (e.g. you skipped the idev check):

```bash
cd $WORK/projects/RAFT-DVC
sbatch scripts/phase1/slurm_smoke_1_2_at64.sh
squeue -u $USER
# when it goes PD -> R:
tail -f logs/smoke_1_2_at64_<jobid>.out
```

**PASS:** banner `cuda True`, params printed, **no OutOfMemoryError**, loss
lines finite, and a `Validation ... EPE:` line prints before the 30 min cap.

**If it OOMs** (only if the GPU is not clean -- a stale process still holding
memory; on a fresh exclusive SLURM node this will not happen): fall back to
`--model-config configs/models/raft_dvc_1_2_p2_r4.yaml --batch-size 1
--grad-accum-steps 8` (non-checkpointed batch 1, ~65 GB; same eff batch 8 and
total_steps 75000 so resume stays valid). Edit BOTH scripts.

Clean up the throwaway smoke dir after:
`rm -rf $SCRATCH/raft-dvc/training_runs/phase1/_smoke_1_2_at64`.

## Step 5 -- Vista: submit the real (resuming) run

```bash
cd $WORK/projects/RAFT-DVC
sbatch scripts/phase1/slurm_1_2_at64_vista.sh
squeue -u $USER
tail -f logs/rdvc_1_2_at64_<jobid>.out
```

Expect the log to print `[resume] .../latest.pth` and continue from epoch ~53
with val EPE already ~1.0 (not the dead-zero plateau). Per-epoch on the GH200
(ckpt batch 4) is ~8.3 min (vs the 5090's ~24 min), so 300 epochs ~= 41 h
(~2 x 24 h windows).

### Crossing the 24 h walltime cap

`--latest-interval 1` writes `latest.pth` every epoch, and the script
auto-resumes from it, so just **resubmit the same script** after each window:

```bash
sbatch scripts/phase1/slurm_1_2_at64_vista.sh          # picks up latest.pth
# or chain automatically so it requeues the instant the prior job ends:
sbatch --dependency=afterany:<prev_jobid> scripts/phase1/slurm_1_2_at64_vista.sh
```

## Step 6 -- confirm healthy, THEN stop the local 5090 run

Vista is "successfully running" once, in `tail -f`, you see: resumed epoch ~53+,
finite decreasing loss, no OOM, and val EPE continuing its downward trend from
~1.0. Only then stop the local run:

```powershell
# on the 5090 (PowerShell)
schtasks /End /TN "<the paper1_axis1_1_2_at64 task name>"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*paper1_axis1_1_2_at64*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Step 7 -- when it reaches epoch 300: pull the checkpoint back

```bash
rsync -avz \
  'zachtong@vista.tacc.utexas.edu:$SCRATCH/raft-dvc/training_runs/phase1/paper1_axis1_1_2_at64/best_model.pth' \
  "/c/Zixiang_local_data/raft-dvc/paper1/phase1/paper1_axis1_1_2_at64/"
```

Then locally: re-run `scripts/paper1/fig_disp_envelope.py` (auto-adds the 1/2
curve) and the 9-cell OOD, and record the final val EPE -- the fm-EPE-law
held-out test predicts it should land near **0.094 input-EPE** (fm-EPE ~0.047).

---

## Gotchas specific to this arm (carried from the 5090 session)

- **Config choice:** use `raft_dvc_1_2_p2_r4_ckpt.yaml` at **batch 4 x accum 2**
  (same config as local, just batch 4 vs 1). Measured on idev: batch 2 non-ckpt
  OOMs the ~96 GB card, but ckpt batch 4 fits (~66 GB) and is ~25% faster than
  non-ckpt batch 1. total_steps stays 75000, so resume is clean.
- **Data determinism:** the val set must match the local val set for the EPE
  curve to be comparable -- Step 2's command guarantees it (same generator seeds).
- **`gh` is 24 h, not unlimited:** the run needs several windows; rely on
  auto-resume (Step 5), never a single job.
- **Don't reuse the stale `slurm_train_otf.sh`** or the `slurm_train.sh`
  template -- they reference old flag names (`--training-config`) and OTF cases,
  not this NPZ arm. Use only the two scripts named in this runbook.
