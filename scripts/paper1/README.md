# Paper-1 Training Scripts

End-to-end training for paper-1's **3-encoder architecture ablation** on a
single bead config (`r4_medium` at `size=64`, input-disp `[0.6, 6]` voxels).

## Cases

| Case | Encoder | Input | fm | Batch | max_lr | 500 epoch time |
|---|---|---|---|---|---|---|
| 1/8 | `raft_dvc_1_8_p4_r4.yaml` | 64³ | 8 | 24 | 4.0e-4 | ~45 min |
| 1/4 | `raft_dvc_1_4_p3_r4.yaml` | 64³ | 16 | 8 | 2.0e-4 | ~7 h |
| 1/2 | `raft_dvc_1_2_p2_r4.yaml` | 64³ | 32 | 1 | 7.0e-5 | ~48 h |

LR uses sqrt-scaling rule: `max_lr = 2e-4 × sqrt(batch / 8)`.

## Prerequisites

1. Generated data at `data_paper1/r4_medium_size64/{train,val,test}/*.npz`
   (see `scripts/generate_phase1_dataset.py` with `--input-disp-range 0.6 6
   --only-config r4_medium --sizes 64`).
2. Same generator runs for the 9 OOD test sets (no `--only-config`, `--test 100`
   only).
3. Conda env active, GPU available, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## How to run

### Option A: each case manually (recommended for first run)

```cmd
scripts\paper1\train_1_8.bat        :: ~45 min, validate pipeline
scripts\paper1\train_1_4.bat        :: ~7 h, run during day with monitoring
scripts\paper1\train_1_2.bat        :: ~48 h, kick off Friday evening
```

### Option B: sequential all-in-one

```cmd
scripts\paper1\train_all_sequential.bat
```

Stops if any case fails (OOM / NaN / crash). Resumable: just re-run the
specific case `.bat` for the one that failed; finished cases skip via
checkpoint naming.

### Option C: trim 1/2 case to 300 epoch (saves ~19 h)

Edit `train_1_2.bat`, change `--epochs 500` to `--epochs 300`. Based on
the 760-epoch convergence study, epoch 300 captures ~85% of asymptotic
EPE improvement.

## Output layout

```
checkpoints/phase1/paper1_1_8_r4md64/
  ├── best_model.pth         <- main deliverable per case
  ├── latest.pth
  ├── checkpoint_epoch_XXXX.pth (every 50 epoch)
  ├── training.log
  ├── run_config.yaml
  └── logs/                   <- TB events

checkpoints/phase1/paper1_1_4_r4md64/  (same layout)
checkpoints/phase1/paper1_1_2_r4md64/  (same layout)
```

## Monitor with TensorBoard

```cmd
:: View all 3 simultaneously
tensorboard --logdir checkpoints\phase1 --port 6006
```

Open browser to <http://localhost:6006>. Filter for `paper1_*` runs.
Key plots: `val/epe` (lower is better), `train/loss_epoch`, `train/lr`.

## After training: cross-config OOD evaluation

Use `scripts/phase1/evaluate_ood.py` to run each trained model on all 9
test configs (see Task #27 in TACC_workflow_runbook.md notes).
