@echo off
REM =============================================================================
REM Paper-1 hypothesis test: 1/4 case with corr_levels=4 (vs failed L=3)
REM =============================================================================
REM Background: 1/4 with L=3 dead-zero across multiple LR/batch/impl tries
REM             (200 epochs each, loss stuck at ~16, val EPE ~2.6).
REM Hypothesis: L=3 provides insufficient coarse-scale lookup signal to
REM             escape predict-zero attractor at random init.  1/8 case
REM             uses L=4 and trains fine.
REM Test:       train 1/4 with L=4 for 50 epochs; if val EPE < 2.0 by epoch
REM             25 -> hypothesis confirmed.  Else dead-zero is unrelated to L.
REM =============================================================================

setlocal

REM ---- Find vcvars (needed for CUDA extension JIT if cache miss) ------------
set "VCVARS="
for %%P in (
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if exist %%P set "VCVARS=%%P"
)
if defined VCVARS (
    echo [vcvars] %VCVARS%
    call %VCVARS% >nul
)
set TORCH_CUDA_ARCH_LIST=12.0

cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v3_1_4_p4_test
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

REM Clean start (no resume) -- this is a fresh hypothesis test
if exist "%EXP_DIR%" (
    echo [reset] Removing previous test dir to start fresh
    rmdir /S /Q "%EXP_DIR%"
)

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM 50 epochs is enough to see clear learning signal (per 1/8 case curve)
python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_4_p4_r4_cuda_otf_v5.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 50 ^
    --batch-size 16 ^
    --max-lr 4.0e-4 ^
    --num-workers 8

if errorlevel 1 (
    echo [ERROR] Test run failed
    exit /b 1
)

echo ====================================================================
echo Hypothesis test done.  Inspect val EPE history:
echo   findstr "Validation" "%EXP_DIR%\training.log"
echo Healthy:   val EPE drops < 2.0 by epoch 25
echo Dead-zero: val EPE stays at ~2.6 throughout
echo ====================================================================

endlocal
