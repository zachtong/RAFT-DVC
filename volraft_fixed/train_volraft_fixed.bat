@echo off
REM =============================================================================
REM Paper-1 baseline: VolRAFT architecture, sampler FIXED, trained on OUR data
REM =============================================================================
REM Released-checkpoint architecture (small encoder: fnet 128, hidden 96,
REM context 64; corr_levels 4, corr_radius 3, iters 12) with the
REM bilinear_sampler axis-order bug fixed (volraft_fixed\model\utils\utils.py).
REM Recipe mirrors paper1_clean_*_r4md64_pw02: r4_medium_size64 [4,8],
REM batch 8 (auto-halves on OOM), 300 epochs, OneCycle 2e-4 pct-start 0.2,
REM seed 42.  Do NOT launch while another training holds the GPU.
REM
REM schtasks-friendly: console output redirected to %EXP_DIR%_console.log.
REM =============================================================================

setlocal
cd /d %~dp0..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set DATA_DIR=G:\Zixiang_local_data\raft-dvc\data_paper1_v3\r4_medium_size64
set EXP_NAME=volraft_fixed_r4md64
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

set RESUME_ARG=
if exist "%EXP_DIR%\latest.pth" (
    echo [resume] Found %EXP_DIR%\latest.pth -- resuming
    set RESUME_ARG=--resume "%EXP_DIR%\latest.pth"
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1

C:\Users\zt3323\.conda\envs\raft-dvc-2\python.exe -u volraft_fixed\train_volraft_fixed.py ^
    --data-dir   "%DATA_DIR%" ^
    --output-dir "%EXP_DIR%" ^
    --epochs 300 ^
    --batch-size 8 ^
    --max-lr 2.0e-4 ^
    --pct-start 0.2 ^
    --num-workers 8 ^
    --seed 42 %RESUME_ARG% > "%EXP_DIR%_console.log" 2>&1

if errorlevel 1 (
    echo [ERROR] Training failed -- see %EXP_DIR%_console.log
    exit /b 1
)

endlocal
