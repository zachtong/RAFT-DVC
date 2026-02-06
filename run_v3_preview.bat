@echo off
REM Temporary script to preview v3 dataset configuration

cd /d "%~dp0"

echo ======================================================================
echo V3 Dataset Preview Generator
echo ======================================================================
echo.
echo This will:
echo   1. Generate 12 preview samples (takes ~3-5 minutes)
echo   2. Create a comprehensive visualization
echo   3. Save to: temp_v3_preview/
echo.
echo Press any key to start...
pause >nul

REM Activate conda environment
call conda activate raft-dvc

REM Run the preview script
python preview_v3_dataset.py

echo.
echo ======================================================================
echo Press any key to exit...
pause >nul
