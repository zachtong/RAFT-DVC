@echo off
REM =============================================================================
REM Paper-1: train 1/4 encoder on r4_medium_size64 (Windows 5090)
REM =============================================================================
REM Mid-size case (~7 h for 500 epoch).  fm=16, same as your earlier validated
REM workload.  batch=8 with standard CorrBlock fits comfortably (~20 GB peak).
REM
REM Output: checkpoints/paper1/1_4_r4md64/best_model.pth
REM =============================================================================

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_4_p3_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1 ^
    --output-root  checkpoints ^
    --experiment-name paper1_1_4_r4md64 ^
    --epochs 500 ^
    --batch-size 8 ^
    --max-lr 2.0e-4 ^
    --num-workers 8
