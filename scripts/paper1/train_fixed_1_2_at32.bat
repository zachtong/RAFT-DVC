@echo off
REM =============================================================================
REM Paper-1 FIXED-sampler retrain, arm 1 of 3: 1/2 encoder @ 32^3 (axis-2 recipe)
REM =============================================================================
REM Identical recipe to paper1_axis2_1_2_at32 (r2_medium_size32, disp [2,4],
REM batch 8, 300 epochs, seed 42).  The ONLY difference is the code fix:
REM corr_sampler_version=2 (bilinear_sampler_3d W<->D axis swap corrected,
REM see tests/test_corr_sampler.py and reports/sampler_bug_diagnosis_2026-07-12).
REM
REM GATE: after this arm finishes, run the size-generalization test BEFORE
REM training arms 2/3 (user-mandated pass criterion: no cliff at 64^3).
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set DATA_ROOT=C:\Zixiang_local_data\raft-dvc\data_paper1_axis2
set EXP_NAME=paper1_fixed_1_2_at32
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

set RESUME_ARG=
if exist "%EXP_DIR%\latest.pth" (
    echo [resume] Found %EXP_DIR%\latest.pth -- resuming
    set RESUME_ARG=--resume "%EXP_DIR%\latest.pth"
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

C:\Users\zt3323\.conda\envs\raft-dvc-2\python.exe scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_2_p2_r4.yaml ^
    --data-config  r2_medium_size32 ^
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
