@echo off
REM =============================================================================
REM Paper-1: train 1/2 encoder on r4_medium_size64 (Windows 5090)
REM =============================================================================
REM SLOWEST case (~48 h for 500 epoch).  fm=32 forces batch=1 on 32 GB GPU.
REM Standard CorrBlock fits (corr volume 1.4 GB + UpdateBlock 20 GB).
REM
REM Considerations:
REM  - batch=1 underutilises GPU compute (cannot fix without exceeding memory).
REM  - max_lr = 7.0e-5 follows sqrt scaling from 2e-4 baseline at batch=8.
REM    Lower LR + small batch => slower convergence; 500 epoch should be enough.
REM  - Consider trimming to 300 epoch (saves ~19 h) if 500 epoch is overkill.
REM    Convergence study showed ~85%% of asymptote captured by epoch 300.
REM
REM Output layout / resume behaviour: see train_1_8.bat header.
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v2_1_2_r4md64
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
    --model-config configs\models\raft_dvc_1_2_p2_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 500 ^
    --batch-size 1 ^
    --max-lr 7.0e-5 ^
    --num-workers 4 %RESUME_ARG%

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
