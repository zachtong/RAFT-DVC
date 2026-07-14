# volraft_fixed -- launch instructions

**Do NOT launch while another training holds the GPU** (a paper-1 run may be
active; check `nvidia-smi` first).

## What this trains

VolRAFT architecture (released `volraft_config_120` variant: small encoder --
fnet 128 / hidden 96 / context 64, corr_levels 4, corr_radius 3, iters 12,
internal per-patch-pair min-max normalization) with the trilinear-sampler
axis-order bug **fixed** (`volraft_fixed/model/utils/utils.py`), trained on
our `r4_medium_size64` NPZ dataset with the paper-1 clean recipe:
300 epochs x 250 optimizer steps (2000 samples / batch 8), AdamW 2e-4,
OneCycleLR pct_start 0.2, fp16 autocast, grad clip 1.0, RAFT sequence loss
gamma 0.8, seed 42.  Batch auto-halves on CUDA OOM (pre-flight probe).

## Exact command (foreground, from repo root)

```bat
cd /d "C:\Users\zt3323\OneDrive - The University of Texas at Austin\Documents\Python Codes\RAFT-DVC"
C:\Users\zt3323\.conda\envs\raft-dvc-2\python.exe -u volraft_fixed\train_volraft_fixed.py ^
    --data-dir   "G:\Zixiang_local_data\raft-dvc\data_paper1_v3\r4_medium_size64" ^
    --output-dir "C:\Zixiang_local_data\raft-dvc\paper1\phase1\volraft_fixed_r4md64" ^
    --epochs 300 --batch-size 8 --max-lr 2.0e-4 --pct-start 0.2 ^
    --num-workers 8 --seed 42
```

Resume after interruption: append
`--resume "C:\Zixiang_local_data\raft-dvc\paper1\phase1\volraft_fixed_r4md64\latest.pth"`
(the .bat does this automatically when `latest.pth` exists).

## Batch launcher (schtasks-friendly, console log redirect built in)

```bat
"C:\Users\zt3323\OneDrive - The University of Texas at Austin\Documents\Python Codes\RAFT-DVC\volraft_fixed\train_volraft_fixed.bat"
```

Console output goes to
`C:\Zixiang_local_data\raft-dvc\paper1\phase1\volraft_fixed_r4md64_console.log`;
the structured `training.log` (plus `latest.pth`, `best_model.pth`,
`checkpoint_epoch_*.pth`, TensorBoard `logs/`, `run_config.yaml`) goes to
`C:\Zixiang_local_data\raft-dvc\paper1\phase1\volraft_fixed_r4md64\`.

### Detached long run via Task Scheduler (survives Claude/RDP session churn)

```bat
schtasks /Create /TN "volraft_fixed_r4md64" /SC ONCE /ST 10:05 /F ^
    /TR "\"C:\Users\zt3323\OneDrive - The University of Texas at Austin\Documents\Python Codes\RAFT-DVC\volraft_fixed\train_volraft_fixed.bat\""
schtasks /Run /TN "volraft_fixed_r4md64"
```

(Adjust `/ST` to a time after the current GPU run finishes, ~10:00.)

## Pre-launch checklist

1. `nvidia-smi` -- no other training on the GPU.
2. `G:\Zixiang_local_data\raft-dvc\data_paper1_v3\r4_medium_size64\{train,val}`
   present (2000 / 200 samples).
3. CPU sanity already verified: `python volraft_fixed\verify_fixed_cpu.py`
   (impulse test on the fixed sampler, corr-chain consistency, tiny 16^3
   forward returning finite flows of the right shape).

## Expected footprint

~250 steps/epoch at batch 8, 64^3.  The upstream config fit batch 18 at
60x80x80 on 32 GB with corr_levels 4; 64^3 has a ~2.4x smaller all-pairs
correlation volume, so batch 8 leaves ample headroom on the 5090 (32 GB).
