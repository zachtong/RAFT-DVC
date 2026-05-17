@echo off
REM =============================================================================
REM Paper-1: train 1/8 encoder on r4_medium_size64 (Windows 5090)
REM =============================================================================
REM Fastest case (~45 min for 500 epoch).  fm=8 corr volume tiny, batch=24 easy.
REM This is the VolRAFT-style baseline architecture for our comparison.
REM
REM Output layout:
REM   Training (live):   C:\Zixiang_local_data\raft-dvc\paper1\phase1\<EXP>\
REM   OneDrive archive:  checkpoints\phase1\<EXP>\  (rsync'd post-training)
REM
REM Why split: tfevents files in OneDrive get locked by sync, killing TB writer
REM thread mid-training.  Local fast SSD avoids this; xcopy at the end pushes
REM everything (best, latest, periodic, logs) into OneDrive for cloud backup.
REM
REM Re-run safety: if latest.pth exists in OUTPUT_ROOT, resume from it.
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v2_1_8_r4md64
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
    --model-config configs\models\raft_dvc_1_8_p4_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 500 ^
    --batch-size 24 ^
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
