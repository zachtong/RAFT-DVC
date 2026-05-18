@echo off
REM =============================================================================
REM Paper-1: train 1/4 encoder on r4_medium_size64 (Windows 5090) -- v5 OTF
REM =============================================================================
REM Architecture: 1/4 encoder + input=64 -> fm=16, corr_levels=3, radius=4
REM
REM Memory strategy (revised 2026-05-18 after prior dead-zero failures):
REM   Std CorrBlock at fm=16 OOMs above batch=10 on 5090 32GB.  Use CUDA OTF v5
REM   (cooperative-tiled multi-warp kernel) to skip storing the corr pyramid,
REM   freeing memory for batch=16.  Closer to 1/8 case's batch=24 for cleaner
REM   ablation than batch=8 std (which the prior attempt failed at).
REM
REM LR: max_lr=4e-4 (same as 1/8 case).  Adam does not require batch scaling.
REM     Earlier sqrt-scaled max_lr=2e-4 caused 200-epoch dead-zero failure --
REM     halved step size left model stuck in init attractor.
REM
REM FALLBACK if batch=16 OOMs:
REM   1. Open this bat in editor.
REM   2. Change "--batch-size 16" to "--batch-size 12".
REM   3. Optionally change EXP_NAME to ..._b12 to keep separate ckpt dirs.
REM   4. Re-run.
REM   (Note: wall-clock at B=12 is similar to B=16 -- per-step faster, more steps.)
REM
REM Output layout / resume behaviour: see train_1_8.bat header.
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v3_1_4_otf_v5_b16
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
    --model-config configs\models\raft_dvc_1_4_p3_r4_cuda_otf_v5.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 500 ^
    --batch-size 16 ^
    --max-lr 4.0e-4 ^
    --num-workers 8 %RESUME_ARG%

if errorlevel 1 (
    echo [ERROR] Training failed -- skipping OneDrive backup
    echo If error was CUDA OOM: edit this bat, change --batch-size 16 to 12, re-run.
    exit /b 1
)

echo ====================================================================
echo Training finished.  Backing up to OneDrive: %ONEDRIVE_DEST%
echo ====================================================================
if not exist "checkpoints\phase1" mkdir "checkpoints\phase1"
xcopy /E /I /Y "%EXP_DIR%" "%ONEDRIVE_DEST%"

endlocal
