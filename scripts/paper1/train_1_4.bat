@echo off
REM =============================================================================
REM Paper-1: train 1/4 encoder on r4_medium_size64 (Windows 5090)
REM =============================================================================
REM Mid-size case (~7 h for 500 epoch).  fm=16, same as your earlier validated
REM workload.  batch=8 with standard CorrBlock fits comfortably (~20 GB peak).
REM
REM LR choice (revised 2026-05-17 after first attempt at max_lr=2e-4 plateaued):
REM   Adam's second-moment normalization makes step size ~ LR independent of
REM   batch.  The conventional sqrt scaling (2e-4 for batch=8 from 4e-4 baseline
REM   at batch=24) is *too conservative* for Adam -- it halves the per-step
REM   movement and leaves the model unable to escape init within OneCycleLR's
REM   short peak window (epochs 25-50).  Empirically: 200 epochs at 2e-4 made
REM   zero progress.
REM   Fix: keep max_lr=4e-4 (same as 1/8 case); 3x more updates per epoch
REM   (150 vs 50) already compensates for batch-noise increase.
REM
REM Output layout / resume behaviour: see train_1_8.bat header.
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v2_1_4_r4md64
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%
set ONEDRIVE_DEST=checkpoints\phase1\%EXP_NAME%

if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

set RESUME_ARG=
if exist "%EXP_DIR%\latest.pth" (
    echo [resume] Found %EXP_DIR%\latest.pth -- resuming
    set RESUME_ARG=--resume "%EXP_DIR%\latest.pth"
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_4_p3_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 500 ^
    --batch-size 8 ^
    --max-lr 4.0e-4 ^
    --num-workers 8 %RESUME_ARG%

if errorlevel 1 (
    echo [ERROR] Training failed -- skipping OneDrive backup
    exit /b 1
)

echo ====================================================================
echo Training finished.  Backing up to OneDrive: %ONEDRIVE_DEST%
echo ====================================================================
if not exist "checkpoints\phase1" mkdir "checkpoints\phase1"
xcopy /E /I /Y "%EXP_DIR%" "%ONEDRIVE_DEST%"

endlocal
