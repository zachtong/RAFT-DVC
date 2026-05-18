@echo off
REM =============================================================================
REM Paper-1 DIAGNOSTIC: 1/4 case 30 epoch with batch=12 (b16 OOM'd)
REM =============================================================================
REM Fallback after b16 OOM on 5090 32GB.  batch=12 should peak at ~28 GB.
REM
REM LR: max_lr=5.0e-4 (sqrt scaling 4e-4 * sqrt(12/8) = 4.9e-4, rounded).
REM
REM If THIS also OOMs -> try batch=10 next (rare given b16 estimate was 36 GB
REM and we're now at 75% of that).
REM
REM AMP was already ruled out by user's --no-amp test.  This isolates batch
REM size as the remaining suspect.
REM
REM Output: C:\Zixiang_local_data\raft-dvc\paper1\phase1\paper1_v2_1_4_DIAG_b12\
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v2_1_4_DIAG_b12
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
    --batch-size 12 ^
    --max-lr 5.0e-4 ^
    --num-workers 8

echo ====================================================================
echo Diag done. Check:
echo   findstr "Train Loss" "%EXP_DIR%\training.log"
echo   findstr "Validation" "%EXP_DIR%\training.log"
echo ====================================================================

endlocal
