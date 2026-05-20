@echo off
REM =============================================================================
REM Paper-1 regression isolation: run 1/8 smoke test on OLD code (aa3c3ca)
REM =============================================================================
REM Purpose: definitively answer "is the dead-zero regression in our recent
REM code changes or somewhere else (env/driver/data)?"
REM
REM Process:
REM   1. Stash any uncommitted local changes
REM   2. Checkout aa3c3ca (last commit before OTF work, when 1/8 v2 trained OK)
REM   3. Run 30-epoch smoke test
REM   4. Restore current branch
REM
REM Expected if our code regressed:  1/8 trains at aa3c3ca, val EPE drops by ep 30
REM Expected if env/data regressed:  1/8 still dead even at aa3c3ca
REM =============================================================================

setlocal
cd /d %~dp0..\..\

echo === Saving current state ===
git stash push -m "before regression test" 2>nul
set "STASH_CREATED=%errorlevel%"

echo === Checking out aa3c3ca (pre-OTF era) ===
git checkout aa3c3ca

set OUTPUT_ROOT=C:\Zixiang_local_data\raft-dvc\paper1
set EXP_NAME=paper1_oldcode_1_8_test
set EXP_DIR=%OUTPUT_ROOT%\phase1\%EXP_NAME%

if not exist "%OUTPUT_ROOT%\phase1" mkdir "%OUTPUT_ROOT%\phase1"
if exist "%EXP_DIR%" rmdir /S /Q "%EXP_DIR%"

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo === Running 30-epoch 1/8 test on OLD code ===
python scripts\phase1\train_phase1.py ^
    --model-config configs\models\raft_dvc_1_8_p4_r4.yaml ^
    --data-config  r4_medium_size64 ^
    --data-root    data_paper1_v2 ^
    --output-root  "%OUTPUT_ROOT%" ^
    --experiment-name %EXP_NAME% ^
    --epochs 30 ^
    --batch-size 24 ^
    --max-lr 4.0e-4 ^
    --num-workers 8

echo === Restoring current branch ===
git checkout master

if "%STASH_CREATED%"=="0" (
    echo === Restoring stashed changes ===
    git stash pop
)

echo.
echo ====================================================================
echo Test done.  Inspect:
echo   findstr "Validation" "%EXP_DIR%\training.log"
echo If OLD code shows val EPE drop by epoch 30 -> regression in our recent changes
echo If OLD code ALSO stays at val EPE ~2.64 -> regression is env/data/driver
echo ====================================================================

endlocal
