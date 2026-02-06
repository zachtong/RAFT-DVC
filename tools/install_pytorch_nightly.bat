@echo off
echo ============================================
echo Installing PyTorch Nightly for RTX 5090
echo ============================================
echo.

echo Step 1: Activating conda environment...
call conda activate raft-dvc
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)

echo.
echo Step 2: Uninstalling current PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 3: Installing PyTorch Nightly (CUDA 12.8 - RTX 5090 support)...
echo   3a. Installing torch...
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
if errorlevel 1 (
    echo ERROR: Failed to install torch
    pause
    exit /b 1
)

echo   3b. Installing torchvision (matching version)...
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
if errorlevel 1 (
    echo WARNING: torchvision installation failed, but torch is installed
)

echo.
echo ============================================
echo Installation complete!
echo ============================================
echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); x = torch.zeros(10).cuda() if torch.cuda.is_available() else torch.zeros(10); print('Test tensor created successfully!')"

echo.
pause
