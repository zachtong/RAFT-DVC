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
REM OTF saves ~2-3 GB of corr+lookup memory (corr_pyramid + checkpointed
REM corr_features across 12 iters), enabling batch=12.  If batch=12 with OTF
REM still produces dead-zero, the issue is NOT primarily about batch.
REM
REM LR: max_lr=4e-4 (same as last batch=8 failure -- keeps LR fixed so we
REM cleanly isolate the batch variable).
REM
REM Epochs: 50 (was 30) -- OneCycleLR warmup ends at epoch 25 (5% of 500),
REM so 30 epoch only sees 5 epoch of peak LR, not enough margin.  50 epoch
REM gives 25 epoch of post-warmup time at near-peak LR.
REM
REM Output: C:\Zixiang_local_data\raft-dvc\paper1\phase1\paper1_v3_1_4_otf_DIAG_b12\
REM =============================================================================

setlocal
cd /d %~dp0..\..\

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_v3_1_4_otf_DIAG_b12
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
    --batch-size 12 ^
    --max-lr 4.0e-4 ^
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
