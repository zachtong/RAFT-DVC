@echo off
REM =============================================================================
REM Paper-1 DIAGNOSTIC: 1/4 case 30 epoch with FP32 (no AMP)
REM =============================================================================
REM Goal: figure out why 1/4 case predicts all zeros (val EPE stuck at flow
REM mean norm = 2.6).  Prime suspect: AMP fp16 grad underflow specific to the
REM 1/4 encoder.
REM
REM This is a SHORT (~30 min) experiment with a different experiment name so
REM it does not collide with the failed full-length run dirs.  If train loss
REM visibly drops in 30 epoch -> AMP was the root cause; if still flat -> try
REM next diagnostic (--seed 43 etc).
REM
REM Reads: data_paper1_v2/r4_medium_size64
REM Writes: C:\Zixiang_local_data\raft-dvc\paper1\phase1\paper1_v2_1_4_DIAG_noamp\
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v2_1_4_DIAG_noamp
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if exist "%EXP_DIR%" (
    echo [reset] Removing previous diag dir to start fresh
    rmdir /S /Q "%EXP_DIR%"
)
if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_4_p3_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 30 ^
    --batch-size 8 ^
    --max-lr 4.0e-4 ^
    --num-workers 8 ^
    --no-amp

echo ====================================================================
echo Diag done. Inspect:
echo   train loss trajectory: type "%EXP_DIR%\training.log" ^| findstr "Train Loss"
echo   val EPE trajectory:    type "%EXP_DIR%\training.log" ^| findstr "Validation"
echo ====================================================================

endlocal
