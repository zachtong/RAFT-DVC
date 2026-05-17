@echo off
REM =============================================================================
REM Paper-1: train 1/8 encoder on r4_medium_size64 (Windows 5090)
REM =============================================================================
REM Fastest case (~45 min for 500 epoch).  fm=8 corr volume tiny, batch=24 easy.
REM This is the VolRAFT-style baseline architecture for our comparison.
REM
REM Output: checkpoints/paper1/1_8_r4md64/best_model.pth
REM =============================================================================

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_8_p4_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1 ^
    --output-root  checkpoints ^
    --experiment-name paper1_1_8_r4md64 ^
    --epochs 500 ^
    --batch-size 24 ^
    --max-lr 4.0e-4 ^
    --num-workers 8
