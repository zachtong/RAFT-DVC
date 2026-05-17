@echo off
REM =============================================================================
REM Paper-1 DIAGNOSTIC: 1/4 case 30 epoch with batch=16 (vs failed batch=8)
REM =============================================================================
REM Hypothesis: batch=8 with 1/4 encoder is too small to escape the "predict
REM zero" attractor.  Adam at small batch behaves like SignSGD with noisy
REM direction -- struggles to find consistent gradient out of flat regions.
REM 1/8 case worked at batch=24 (3x bigger), so try doubling batch on 1/4.
REM
REM LR: max_lr=5.6e-4 (sqrt scaling from 4e-4 at batch=8 -- but tempered by
REM "Adam doesn't strictly need batch scaling"; this is a middle ground).
REM
REM Memory: estimated ~36 GB on 5090 (32 GB) -- MIGHT OOM.  If OOM, fall back
REM to diag_1_4_b12.bat (TBD).
REM
REM Output: C:\Zixiang_local_data\raft-dvc\paper1\phase1\paper1_v2_1_4_DIAG_b16\
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v2_1_4_DIAG_b16
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if exist "%EXP_DIR%" (
    echo [reset] Removing previous diag dir
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
    --batch-size 16 ^
    --max-lr 5.6e-4 ^
    --num-workers 8

echo ====================================================================
echo Diag done. Check:
echo   findstr "Train Loss" "%EXP_DIR%\training.log"
echo   findstr "Validation" "%EXP_DIR%\training.log"
echo ====================================================================

endlocal
