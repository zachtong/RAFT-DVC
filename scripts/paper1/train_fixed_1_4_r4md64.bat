@echo off
REM =============================================================================
REM Paper-1 FIXED-sampler retrain, arm 2 of 3: 1/4 encoder @ 64^3
REM =============================================================================
REM Identical recipe to paper1_clean_1_4_r4md64_pw02 (r4_medium_size64 [4,8],
REM batch 8, 300 epochs, pct-start 0.2, seed 42); only difference is the
REM corr_sampler_version=2 code fix.  Launch ONLY after the arm-1 size-
REM generalization gate passes.
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set DATA_ROOT=C:\Zixiang_local_data\raft-dvc\data_paper1_v3
set EXP_NAME=paper1_fixed_1_4_r4md64
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

set RESUME_ARG=
if exist "%EXP_DIR%\latest.pth" (
    echo [resume] Found %EXP_DIR%\latest.pth -- resuming
    set RESUME_ARG=--resume "%EXP_DIR%\latest.pth"
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

C:\Users\zt3323\.conda\envs\raft-dvc-2\python.exe scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_4_p2_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    "%DATA_ROOT%" ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 300 ^
    --batch-size 8 ^
    --max-lr 2.0e-4 ^
    --pct-start 0.2 ^
    --num-workers 8 ^
    --seed 42 %RESUME_ARG%

endlocal
