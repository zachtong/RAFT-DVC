# Move the 1/2 @ input-64 (fm32) arm from the RTX-5090 to TACC Vista

**Date:** 2026-07-09
**Goal:** Finish `paper1_axis1_1_2_at64` (the fm32 arm that completes the Axis-1
3-point curve) on Vista GH200, **resuming** the local epoch-~52 checkpoint, so
the 5090 can be freed for other ML work. Confirm the Vista run is healthy
*before* stopping the local one.

**Why this is clean:** gradient checkpointing (`checkpoint_updates` /
`checkpoint_corr` in the local `*_ckpt` config) only changes how the *backward*
pass trades compute for memory -- it does **not** change the model weights,
optimizer state, or LR schedule. Vista has 96 GB HBM3, so the fm32 correlation
fits **without** checkpointing at batch 2. Using batch 2 x accum 4 keeps the
OneCycle `total_steps` identical to the local batch 1 x accum 8 run
(300 * (2000/bs // accum) = 300 * 250 = **75000** either way), so the copied
checkpoint's scheduler state resumes on the exact same LR curve. Net effect:
same run, faster hardware, no checkpointing overhead.

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

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
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

## Step 4 -- Vista: smoke test (gh-dev, 30 min, fresh, verifies batch 2 fits)

```bash
cd $WORK/projects/RAFT-DVC
sbatch scripts/phase1/slurm_smoke_1_2_at64.sh
squeue -u $USER
# when it goes PD -> R:
tail -f logs/smoke_1_2_at64_<jobid>.out
```

**PASS:** banner `cuda True`, params printed, **no OutOfMemoryError**, loss
lines finite, and a `Validation ... EPE:` line prints before the 30 min cap.

**If it OOMs** (batch 2 fm32 > 96 GB -- unlikely, est. ~62 GB): edit BOTH
`slurm_smoke_1_2_at64.sh` and `slurm_1_2_at64_vista.sh` to `--batch-size 1
--grad-accum-steps 8` (still non-checkpointed, ~31 GB, and `total_steps` still
75000 so resume stays valid), then re-smoke.

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
with val EPE already ~1.0 (not the dead-zero plateau). Per-epoch on H100
(batch 2, non-checkpointed) should be well under the 5090's ~24 min.

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

- **Config choice:** use `raft_dvc_1_2_p2_r4.yaml` on Vista (standard corr, no
  checkpointing) -- NOT the local `raft_dvc_1_2_p2_r4_ckpt.yaml`. Checkpointing
  is a 32 GB-card workaround; on 96 GB it only slows you down. Resume across the
  config swap is valid (weights/optimizer/scheduler are identical either way).
- **Data determinism:** the val set must match the local val set for the EPE
  curve to be comparable -- Step 2's command guarantees it (same generator seeds).
- **`gh` is 24 h, not unlimited:** the run needs several windows; rely on
  auto-resume (Step 5), never a single job.
- **Don't reuse the stale `slurm_train_otf.sh`** or the `slurm_train.sh`
  template -- they reference old flag names (`--training-config`) and OTF cases,
  not this NPZ arm. Use only the two scripts named in this runbook.
