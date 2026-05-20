@echo off
REM =============================================================================
REM Paper-1 sanity check: re-train 1/8 case under CURRENT codebase
REM =============================================================================
REM Purpose: rule out the possibility that recent codebase changes (trainer
REM resume fix, OTF kernel additions, etc.) broke the trainer for ALL cases,
REM not just 1/4.  The original 1/8 v2 ckpt (val EPE 0.43) was trained
REM BEFORE these changes, so doesn't directly verify current code.
REM
REM Setup: matches the original successful 1/8 v2 setup exactly --
REM   encoder: 1/8 (BasicEncoder, corr_levels=4)
REM   data:    r4_medium_size64 (paper-1 v2 dataset)
REM   batch:   24
REM   max_lr:  4e-4
REM   corr_impl: standard
REM   epochs:  50 (smoke test, original ran 500)
REM   seed:    42
REM
REM Pass criteria: val EPE < 1.2 by epoch 100 (original hit 0.43 at epoch 495).
REM Fail criteria: val EPE stays > 2.0 by epoch 100 -> codebase broken.
REM
REM Note: 50 epoch was NOT enough -- LR schedule (OneCycleLR pct_start=0.05)
REM gives only ~2.5 epoch warmup + 47.5 epoch cosine anneal at 50 epoch
REM total.  Need 100-200 epoch for the model to actually escape init and
REM converge meaningfully.  Increased to 200 epoch (~40 min on 5090).
REM
REM Resume note: OneCycleLR cannot be safely resumed beyond its original
REM total_steps.  If you want a longer run, RESTART from scratch (this bat
REM auto-removes any previous ckpt dir).
REM
REM Estimated wall-clock: ~40 min for 200 epochs (1/8 case is fast at fm=8).
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v3_1_8_smoke_retest
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

REM Fresh start (no resume) -- this is a sanity check
if exist "%EXP_DIR%" (
    echo [reset] Removing previous test dir to start fresh
    rmdir /S /Q "%EXP_DIR%"
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_8_p4_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 200 ^
    --batch-size 24 ^
    --max-lr 4.0e-4 ^
    --num-workers 8

if errorlevel 1 (
    echo [ERROR] Smoke test failed -- this is a serious problem since 1/8 should work
    exit /b 1
)

echo ====================================================================
echo Smoke test done.  Inspect val EPE history:
echo   findstr "Validation" "%EXP_DIR%\training.log"
echo Healthy:   val EPE drops below 1.2 by epoch 100 (toward ~0.5 by 200)
echo Broken:    val EPE stays above 2.0 -> trainer/data broken (not 1/4 specific)
echo ====================================================================

endlocal
