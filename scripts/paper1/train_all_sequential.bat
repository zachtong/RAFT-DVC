@echo off
REM =============================================================================
REM Paper-1: run all 3 trainings sequentially on 5090.
REM Total wall-clock estimate: ~56 hours (~2.5 days)
REM Breakdown:
REM   1/8 (b=24, easiest):  ~45 min
REM   1/4 (b=8, mid):       ~7 h
REM   1/2 (b=1, hardest):   ~48 h  <-- consider running this one alone overnight
REM =============================================================================
REM
REM Recommended invocation:
REM   - Best:  run train_1_8.bat + train_1_4.bat back-to-back during day
REM            (~8 h, can monitor), then start train_1_2.bat for 2-day overnight.
REM   - All-in-one: call this script.  Fails fast if any case OOMs / errors.
REM
REM =============================================================================

setlocal
cd /d %~dp0..\..\

echo ====================================================================
echo Paper-1 sequential training start: %DATE% %TIME%
echo ====================================================================

call scripts\paper1\train_1_8.bat
if errorlevel 1 (
    echo [ABORT] 1/8 case failed
    exit /b 1
)
echo --- 1/8 done at %DATE% %TIME% ---

call scripts\paper1\train_1_4.bat
if errorlevel 1 (
    echo [ABORT] 1/4 case failed
    exit /b 1
)
echo --- 1/4 done at %DATE% %TIME% ---

call scripts\paper1\train_1_2.bat
if errorlevel 1 (
    echo [ABORT] 1/2 case failed
    exit /b 1
)

echo ====================================================================
echo All 3 paper-1 trainings completed: %DATE% %TIME%
echo Checkpoints at: checkpoints\phase1\paper1_*
echo ====================================================================
endlocal
