@echo off
REM =============================================================================
REM Paper-1 DIAGNOSTIC: 1/4 case 30 epoch with OTF + batch=12
REM =============================================================================
REM After:
REM   - batch=8 fp16 max_lr=2e-4 -> failed (val EPE 2.6)
REM   - batch=8 fp16 max_lr=4e-4 -> failed (val EPE 2.6)
REM   - batch=8 fp32 (--no-amp)  -> failed (val EPE 2.6) [user confirmed]
REM   - batch=16 -> OOM
REM   - batch=12 -> OOM
REM
REM Hypothesis: batch=8 is too small to escape the "predict zero" attractor.
REM OTF saves ~45% of memory at fm=16 (verified by profile_memory.py: 26 GB
REM std vs 14 GB OTF at batch=8), enabling batch=16 to fit within 5090 32GB.
REM batch=16 is 2x of failed batch=8 and 2/3 of working 1/8 batch=24 -- much
REM stronger test of the "small batch causes dead-zero" hypothesis.
REM
REM LR: max_lr=5.7e-4 = 4e-4 * sqrt(16/8) (sqrt scaling from prior failure).
REM
REM Epochs: 50 -- OneCycleLR warmup ends at epoch 25; 50 epoch gives 25 epoch
REM of near-peak LR window to escape init.
REM
REM Output: C:\Zixiang_local_data\raft-dvc\paper1\phase1\paper1_v3_1_4_otf_DIAG_b16\
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v3_1_4_otf_DIAG_b16
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if exist "%EXP_DIR%" (
    echo [reset] Removing previous diag dir
    rmdir /S /Q "%EXP_DIR%"
)
if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_4_p3_r4_otf.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 50 ^
    --batch-size 16 ^
    --max-lr 5.7e-4 ^
    --num-workers 8

echo ====================================================================
echo Diag done.  Inspect train loss + val EPE:
echo   findstr "Train Loss" "%EXP_DIR%\training.log"
echo   findstr "Validation" "%EXP_DIR%\training.log"
echo.
echo If train loss visibly drops AND val EPE escapes 2.6 -> OTF + b12 works,
echo proceed to full 500-epoch run.
echo If still dead at val EPE ~ 2.6 -> batch is NOT the primary cause; need
echo deeper debug (seed, architecture inspect).
echo ====================================================================

endlocal
