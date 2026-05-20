@echo off
REM =============================================================================
REM Paper-1 sanity check: 1/8 case at 500 epoch (matching original success)
REM =============================================================================
REM Discovery (2026-05-20): 200 epoch retest of 1/8 dead-zero at val EPE 2.64.
REM Cause: OneCycleLR with epochs=200 (vs original epochs=500) gives only
REM ~1500 high-LR steps vs original's ~3750.  Compressed schedule = less
REM peak-LR exposure -> insufficient to escape predict-zero attractor.
REM
REM This bat re-runs with epochs=500 to match the original successful setup
REM exactly.  If this succeeds (val EPE drops by ep 30 like original), the
REM "regression" was just the schedule.  If still dead, real regression exists.
REM
REM Estimated wall-clock: ~3 hours for 500 epochs at batch=24 fm=8.
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v3_1_8_verify_500ep
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

REM Fresh start
if exist "%EXP_DIR%" (
    echo [reset] Removing previous test dir
    rmdir /S /Q "%EXP_DIR%"
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_8_p4_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 500 ^
    --batch-size 24 ^
    --max-lr 4.0e-4 ^
    --num-workers 8

if errorlevel 1 (
    echo [ERROR] Verification failed
    exit /b 1
)

echo ====================================================================
echo Verification done.  Inspect val EPE trajectory:
echo   findstr "Validation" "%EXP_DIR%\training.log"
echo Healthy:  val EPE drops below 2.4 by epoch 30 (matching original)
echo Real bug: still dead-zero at val EPE 2.64 throughout
echo ====================================================================

endlocal
