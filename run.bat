@echo off
setlocal enabledelayedexpansion
REM ======================================================================
REM RAFT-DVC Unified Entry Point
REM Main script for training, testing, and data generation
REM ======================================================================

REM Fix OpenMP duplicate library warning
set KMP_DUPLICATE_LIB_OK=TRUE

:MAIN_MENU
cls
echo ======================================================================
echo                    RAFT-DVC - Main Menu
echo ======================================================================
echo.
echo  1. Training
echo  2. Inference / Testing
echo  3. Dataset Generation
echo  4. Test Set Evaluation
echo  5. Lehu Dataset Testing
echo  6. Exit
echo.
echo ======================================================================
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto TRAINING_MENU
if "%choice%"=="2" goto INFERENCE_MENU
if "%choice%"=="3" goto DATASET_MENU
if "%choice%"=="4" goto TESTSET_MENU
if "%choice%"=="5" goto LEHU_MENU
if "%choice%"=="6" goto EXIT
echo Invalid choice! Press any key to try again...
pause >nul
goto MAIN_MENU

REM ======================================================================
REM TRAINING MENU
REM ======================================================================
:TRAINING_MENU
cls
echo ======================================================================
echo                    Training Menu
echo ======================================================================
echo.
echo  1. Train new model (default config)
echo  2. Train with custom config
echo  3. Resume training from checkpoint
echo  4. Back to main menu
echo.
echo ======================================================================
set /p train_choice="Enter your choice (1-4): "

if "%train_choice%"=="1" goto TRAIN_NEW
if "%train_choice%"=="2" goto TRAIN_CUSTOM
if "%train_choice%"=="3" goto TRAIN_RESUME
if "%train_choice%"=="4" goto MAIN_MENU
echo Invalid choice! Press any key to try again...
pause >nul
goto TRAINING_MENU

:TRAIN_NEW
cls
echo ======================================================================
echo Training New Model (Default Configuration)
echo ======================================================================
echo.
echo Model:    1/8 resolution encoder (baseline)
echo Training: 300 epochs, batch_size=5
echo Dataset:  data/synthetic_confocal/
echo Output:   outputs/training/confocal_baseline/
echo.
echo Estimated time: 8-12 hours on RTX 5090
echo.
pause

call conda activate raft-dvc-2

echo.
echo Starting training with default configs...
echo.

python scripts/train_confocal.py

echo.
echo ======================================================================
echo Training complete! Check outputs/training/confocal_baseline/
echo ======================================================================
pause
goto MAIN_MENU

:TRAIN_CUSTOM
cls
echo ======================================================================
echo Step 1/2: Select Model Architecture
echo ======================================================================
echo.
echo Available model configurations:
echo.

REM Dynamically list all model configs
set model_idx=0
for %%F in (configs\models\*.yaml) do (
    set /a model_idx+=1
    set "model_file_!model_idx!=%%F"
    echo  !model_idx!. %%~nxF
)

echo.
echo  0. Back to menu
echo.
echo ======================================================================
set /p model_choice="Select model config (0-%model_idx%): "

if "%model_choice%"=="0" goto TRAINING_MENU

if "%model_choice%"=="" (
    echo ERROR: No selection made
    pause
    goto TRAINING_MENU
)

REM Validate choice and get selected file
if %model_choice% gtr %model_idx% (
    echo ERROR: Invalid selection
    pause
    goto TRAIN_CUSTOM
)

if %model_choice% lss 1 (
    echo ERROR: Invalid selection
    pause
    goto TRAIN_CUSTOM
)

set "model_config=!model_file_%model_choice%!"

if not exist "%model_config%" (
    echo ERROR: Model config not found: %model_config%
    pause
    goto TRAINING_MENU
)

REM Step 2: Training config
:TRAIN_SELECT_TRAINING
cls
echo ======================================================================
echo Step 2/2: Select Training Configuration
echo ======================================================================
echo.
echo Model selected: %model_config%
echo.
echo Available training configurations:
echo.

REM Dynamically list all training configs
set training_idx=0
for %%F in (configs\training\*.yaml) do (
    set /a training_idx+=1
    set "training_file_!training_idx!=%%F"
    echo  !training_idx!. %%~nxF
)

set /a custom_idx=training_idx+1
set /a back_idx=training_idx+2

echo.
echo  %custom_idx%. Enter custom path
echo  %back_idx%. Back to model selection
echo.
echo ======================================================================
set /p training_choice="Select training config (1-%back_idx%): "

if "%training_choice%"=="%custom_idx%" goto TRAIN_CUSTOM_TRAINING_PATH
if "%training_choice%"=="%back_idx%" goto TRAIN_CUSTOM

if "%training_choice%"=="" (
    echo ERROR: No selection made
    pause
    goto TRAIN_SELECT_TRAINING
)

REM Validate choice and get selected file
if %training_choice% gtr %back_idx% (
    echo ERROR: Invalid selection
    pause
    goto TRAIN_SELECT_TRAINING
)

if %training_choice% lss 1 (
    echo ERROR: Invalid selection
    pause
    goto TRAIN_SELECT_TRAINING
)

set "training_config=!training_file_%training_choice%!"

goto TRAIN_VALIDATE_CONFIG

:TRAIN_CUSTOM_TRAINING_PATH
echo.
set /p training_config="Enter training config path (e.g., configs/training/my_config.yaml): "
if "%training_config%"=="" (
    echo ERROR: No path specified
    pause
    goto TRAIN_SELECT_TRAINING
)

:TRAIN_VALIDATE_CONFIG
if not exist "%training_config%" (
    echo ERROR: Training config not found: %training_config%
    pause
    goto TRAIN_SELECT_TRAINING
)

REM Execute training
cls
echo ======================================================================
echo Starting Training
echo ======================================================================
echo.
echo Model:    %model_config%
echo Training: %training_config%
echo.
pause

call conda activate raft-dvc-2

echo.
python scripts/train_confocal.py --model-config %model_config% --training-config %training_config%

echo.
echo ======================================================================
echo Training complete!
echo ======================================================================
pause
goto MAIN_MENU

:TRAIN_RESUME
cls
echo ======================================================================
echo Resume Training from Checkpoint
echo ======================================================================
echo.
set /p checkpoint_path="Enter checkpoint path: "
if "%checkpoint_path%"=="" (
    echo ERROR: No checkpoint specified
    pause
    goto TRAINING_MENU
)

echo.
echo Use default configs? (y/n)
echo   Default: 1/8 model + baseline training
echo.
set /p use_default="Your choice (y/n): "

if /i "%use_default%"=="y" (
    set model_config=configs/models/raft_dvc_1_8_p4_r4.yaml
    set training_config=configs/training/confocal_baseline.yaml
    goto RESUME_TRAINING
)

REM Custom model selection
echo.
echo Select model config:
set model_idx=0
for %%F in (configs\models\*.yaml) do (
    set /a model_idx+=1
    set "model_file_!model_idx!=%%F"
    echo  !model_idx!. %%~nxF
)
set /p model_choice="Choice (1-%model_idx%): "
set "model_config=!model_file_%model_choice%!"

echo.
echo Select training config:
set training_idx=0
for %%F in (configs\training\*.yaml) do (
    set /a training_idx+=1
    set "training_file_!training_idx!=%%F"
    echo  !training_idx!. %%~nxF
)
set /p training_choice="Choice (1-%training_idx%): "
set "training_config=!training_file_%training_choice%!"

:RESUME_TRAINING
call conda activate raft-dvc-2

echo.
echo Resuming from: %checkpoint_path%
echo Model:    %model_config%
echo Training: %training_config%
echo.

python scripts/train_confocal.py --model-config %model_config% --training-config %training_config% --resume %checkpoint_path%

echo.
echo ======================================================================
echo Training complete!
echo ======================================================================
pause
goto MAIN_MENU

REM ======================================================================
REM INFERENCE MENU
REM ======================================================================
:INFERENCE_MENU
cls
echo ======================================================================
echo                    Inference / Testing Menu
echo ======================================================================
echo.
echo  1. Select checkpoint from training outputs
echo  2. Enter custom checkpoint path
echo  3. Back to main menu
echo.
echo ======================================================================
set /p infer_choice="Enter your choice (1-3): "

if "%infer_choice%"=="1" goto SELECT_CHECKPOINT
if "%infer_choice%"=="2" goto CUSTOM_CHECKPOINT_PATH
if "%infer_choice%"=="3" goto MAIN_MENU
echo Invalid choice! Press any key to try again...
pause >nul
goto INFERENCE_MENU

:SELECT_CHECKPOINT
cls
echo ======================================================================
echo Select Checkpoint
echo ======================================================================
echo.
echo Scanning outputs/training/ for available checkpoints...
echo.

REM Check if outputs/training directory exists
if not exist "outputs\training\" (
    echo ERROR: No training outputs found at outputs/training/
    echo.
    echo Please train a model first or use custom checkpoint path.
    pause
    goto INFERENCE_MENU
)

REM Dynamically list all .pth files (sorted by time, newest first)
REM First collect checkpoint_best.pth files
set ckpt_idx=0
for /f "delims=" %%F in ('dir /s /b /o:-d "outputs\training\checkpoint_best.pth" 2^>nul') do (
    set /a ckpt_idx+=1
    set "ckpt_file_!ckpt_idx!=%%F"
    REM Display relative path
    set "filepath=%%F"
    setlocal enabledelayedexpansion
    set "relpath=!filepath:%CD%\=!"
    echo  !ckpt_idx!. !relpath! ^<-- BEST
    endlocal
)

REM Then list epoch checkpoints (sorted by time, newest first)
for /f "delims=" %%F in ('dir /s /b /o:-d "outputs\training\checkpoint_epoch_*.pth" 2^>nul') do (
    set /a ckpt_idx+=1
    set "ckpt_file_!ckpt_idx!=%%F"
    set "filepath=%%F"
    setlocal enabledelayedexpansion
    set "relpath=!filepath:%CD%\=!"
    echo  !ckpt_idx!. !relpath!
    endlocal
)

if %ckpt_idx%==0 (
    echo No checkpoints found in outputs/training/
    echo.
    echo Please train a model first or use custom checkpoint path.
    pause
    goto INFERENCE_MENU
)

set /a custom_idx=ckpt_idx+1
set /a back_idx=ckpt_idx+2

echo.
echo  %custom_idx%. Enter custom checkpoint path
echo  %back_idx%. Back to inference menu
echo.
echo ======================================================================
set /p ckpt_choice="Select checkpoint (1-%back_idx%): "

if "%ckpt_choice%"=="%custom_idx%" goto CUSTOM_CHECKPOINT_PATH
if "%ckpt_choice%"=="%back_idx%" goto INFERENCE_MENU

if "%ckpt_choice%"=="" (
    echo ERROR: No selection made
    pause
    goto SELECT_CHECKPOINT
)

REM Validate choice
if %ckpt_choice% gtr %back_idx% (
    echo ERROR: Invalid selection
    pause
    goto SELECT_CHECKPOINT
)

if %ckpt_choice% lss 1 (
    echo ERROR: Invalid selection
    pause
    goto SELECT_CHECKPOINT
)

set "checkpoint=!ckpt_file_%ckpt_choice%!"

REM Verify checkpoint exists
if not exist "%checkpoint%" (
    echo ERROR: Checkpoint not found: %checkpoint%
    pause
    goto SELECT_CHECKPOINT
)

goto SELECT_DATASET

:CUSTOM_CHECKPOINT_PATH
echo.
set /p checkpoint="Enter checkpoint path (e.g., outputs/training/confocal_small_1_1/checkpoint_best.pth): "

if "%checkpoint%"=="" (
    echo ERROR: No path specified
    pause
    goto INFERENCE_MENU
)

if not exist "%checkpoint%" (
    echo ERROR: Checkpoint not found: %checkpoint%
    pause
    goto CUSTOM_CHECKPOINT_PATH
)

goto SELECT_DATASET

:SELECT_DATASET
cls
echo ======================================================================
echo Select Dataset
echo ======================================================================
echo.
echo Checkpoint: %checkpoint%
echo.

REM Check for available datasets in data/ directory
echo Available datasets:
echo.

set dataset_idx=0

REM Scan for dataset directories containing train/val/test subdirs
for /d %%D in (data\*) do (
    REM Check if directory has at least one of train/val/test subdirs
    set valid=0
    if exist "%%D\train\" set valid=1
    if exist "%%D\val\" set valid=1
    if exist "%%D\test\" set valid=1

    if !valid!==1 (
        set /a dataset_idx+=1
        set "dataset_path_!dataset_idx!=%%D"
        set "dataset_name_!dataset_idx!=%%~nxD"
        echo  !dataset_idx!. %%~nxD
    )
)

if %dataset_idx%==0 (
    echo ERROR: No valid datasets found in data/ directory
    echo.
    echo A valid dataset must have train/val/test subdirectories.
    pause
    goto SELECT_CHECKPOINT
)

set /a custom_idx=dataset_idx+1
set /a back_idx=dataset_idx+2

echo.
echo  %custom_idx%. Enter custom dataset path
echo  %back_idx%. Back to checkpoint selection
echo.
echo ======================================================================
set /p dataset_choice="Select dataset (1-%back_idx%): "

if "%dataset_choice%"=="%custom_idx%" goto CUSTOM_DATASET_PATH
if "%dataset_choice%"=="%back_idx%" goto SELECT_CHECKPOINT

if "%dataset_choice%"=="" (
    echo ERROR: No selection made
    pause
    goto SELECT_DATASET
)

if %dataset_choice% gtr %back_idx% (
    echo ERROR: Invalid selection
    pause
    goto SELECT_DATASET
)

if %dataset_choice% lss 1 (
    echo ERROR: Invalid selection
    pause
    goto SELECT_DATASET
)

set "data_dir=!dataset_path_%dataset_choice%!"
set "dataset_name=!dataset_name_%dataset_choice%!"

goto SELECT_MODEL_CONFIG

:CUSTOM_DATASET_PATH
echo.
set /p data_dir="Enter dataset path (e.g., data/synthetic_confocal_small): "

if "%data_dir%"=="" (
    echo ERROR: No path specified
    pause
    goto SELECT_DATASET
)

REM Check if dataset has at least one split directory
set valid=0
if exist "%data_dir%\train\" set valid=1
if exist "%data_dir%\val\" set valid=1
if exist "%data_dir%\test\" set valid=1

if !valid!==0 (
    echo ERROR: Invalid dataset - no train/val/test subdirectories found
    pause
    goto CUSTOM_DATASET_PATH
)

REM Extract dataset name from path
for %%D in ("%data_dir%") do set "dataset_name=%%~nxD"

goto SELECT_MODEL_CONFIG

:SELECT_MODEL_CONFIG
cls
echo ======================================================================
echo Select Model Configuration
echo ======================================================================
echo.
echo Checkpoint: %checkpoint%
echo Dataset:    %data_dir%
echo.
echo Available model configurations in configs/models/:
echo.

REM Collect all .yaml files in configs/models/
set model_idx=0
for %%F in (configs\models\*.yaml) do (
    set /a model_idx+=1
    set "model_file_!model_idx!=%%F"
    echo  !model_idx!^. %%~nxF
)

if !model_idx!==0 (
    echo WARNING: No model config files found in configs/models/
    echo.
    echo The inference script will try to load config from checkpoint.
    echo If that fails, it will use default config ^(1/8 encoder, r=4^).
    echo.
    pause
    set "model_config="
    goto SELECT_SAMPLE
)

echo.
echo  0. Skip ^(auto-detect from checkpoint or use defaults^)
echo.
echo ======================================================================
set /p model_choice="Select model config (0-!model_idx!): "

if "%model_choice%"=="0" (
    set "model_config="
    goto SELECT_SAMPLE
)

if "%model_choice%"=="" (
    echo ERROR: No selection made
    pause
    goto SELECT_MODEL_CONFIG
)

if %model_choice% gtr !model_idx! (
    echo ERROR: Invalid selection
    pause
    goto SELECT_MODEL_CONFIG
)

if %model_choice% lss 0 (
    echo ERROR: Invalid selection
    pause
    goto SELECT_MODEL_CONFIG
)

set "model_config=!model_file_%model_choice%!"
echo.
echo Selected: %model_config%
echo.
goto SELECT_SPLIT

:SELECT_SPLIT
cls
echo ======================================================================
echo Select Dataset Split
echo ======================================================================
echo.
echo Checkpoint:    %checkpoint%
echo Dataset:       %data_dir%
echo Model Config:  %model_config%
echo.
echo Options:
echo  1. Validation set (val)
echo  2. Test set (test)
echo  3. Training set (train)
echo  4. Back to model config selection
echo.
echo ======================================================================
set /p split_choice="Enter your choice (1-4): "

if "%split_choice%"=="1" (
    set split=val
    goto SELECT_INFERENCE_MODE
)

if "%split_choice%"=="2" (
    set split=test
    goto SELECT_INFERENCE_MODE
)

if "%split_choice%"=="3" (
    set split=train
    goto SELECT_INFERENCE_MODE
)

if "%split_choice%"=="4" goto SELECT_MODEL_CONFIG

echo Invalid choice! Press any key to try again...
pause >nul
goto SELECT_SPLIT

:SELECT_INFERENCE_MODE
cls
echo ======================================================================
echo Select Inference Mode
echo ======================================================================
echo.
echo Checkpoint:    %checkpoint%
echo Dataset:       %data_dir%
echo Model Config:  %model_config%
echo Split:         %split%
echo.
echo Options:
echo  1. Single sample inference
echo  2. Batch inference (all samples)
echo  3. Back to split selection
echo.
echo ======================================================================
set /p mode_choice="Enter your choice (1-3): "

if "%mode_choice%"=="1" (
    set inference_mode=single
    goto SELECT_SAMPLE
)

if "%mode_choice%"=="2" (
    set inference_mode=batch
    goto SELECT_NUM_VIS
)

if "%mode_choice%"=="3" goto SELECT_SPLIT

echo Invalid choice! Press any key to try again...
pause >nul
goto SELECT_INFERENCE_MODE

:SELECT_NUM_VIS
cls
echo ======================================================================
echo Select Number of Visualizations
echo ======================================================================
echo.
echo Checkpoint:    %checkpoint%
echo Dataset:       %data_dir%
echo Model Config:  %model_config%
echo Split:         %split%
echo Mode:          Batch (all samples)
echo.
echo Options:
echo  1. First 3 samples (default)
echo  2. First 10 samples
echo  3. All samples (may take a while!)
echo  4. Custom number
echo  5. Back to mode selection
echo.
echo ======================================================================
set /p vis_choice="Enter your choice (1-5): "

if "%vis_choice%"=="1" (
    set num_vis=3
    goto SELECT_3D_RENDERING
)

if "%vis_choice%"=="2" (
    set num_vis=10
    goto SELECT_3D_RENDERING
)

if "%vis_choice%"=="3" (
    set num_vis=-1
    goto SELECT_3D_RENDERING
)

if "%vis_choice%"=="4" (
    echo.
    set /p num_vis="Enter number of samples to visualize: "
    if "!num_vis!"=="" set num_vis=3
    goto SELECT_3D_RENDERING
)

if "%vis_choice%"=="5" goto SELECT_INFERENCE_MODE

echo Invalid choice! Press any key to try again...
pause >nul
goto SELECT_NUM_VIS

:SELECT_SAMPLE
cls
echo ======================================================================
echo Select Sample to Test
echo ======================================================================
echo.
echo Checkpoint:    %checkpoint%
echo Dataset:       %data_dir%
echo Model Config:  %model_config%
echo Split:         %split%
echo Mode:          Single sample
echo.
echo Options:
echo  1. Default sample (index 90)
echo  2. Enter custom sample index
echo  3. Back to inference mode selection
echo.
echo ======================================================================
set /p sample_choice="Enter your choice (1-3): "

if "%sample_choice%"=="1" (
    set sample=90
    goto SELECT_3D_RENDERING
)

if "%sample_choice%"=="2" (
    echo.
    set /p sample="Enter sample index: "
    if "!sample!"=="" set sample=90
    goto SELECT_3D_RENDERING
)

if "%sample_choice%"=="3" goto SELECT_INFERENCE_MODE

echo Invalid choice! Press any key to try again...
pause >nul
goto SELECT_SAMPLE

:SELECT_3D_RENDERING
cls
echo ======================================================================
echo Enable 3D Volume Rendering?
echo ======================================================================
echo.
echo Checkpoint:    %checkpoint%
echo Dataset:       %data_dir%
echo Split:         %split%
if "%inference_mode%"=="single" (
    echo Mode:          Single sample ^(%sample%^)
) else (
    echo Mode:          Batch ^(%num_vis% samples^)
)
echo.
echo 3D Volume Rendering generates high-quality side-by-side comparison:
echo   - Left:  Uncertainty volume ^(red, from model prediction^)
echo   - Right: Feature density volume ^(green^)
echo   - Optional: 2D slice visualization below 3D volumes
echo.
echo Note: Only works for models with uncertainty head!
echo       Uses same parameters as demo_side_by_side.py
echo.
echo Options:
echo  1. Yes - Enable 3D rendering ^(high quality, slower^)
echo  2. Yes - Enable 3D rendering with slice visualization
echo  3. No  - Skip 3D rendering ^(faster, default^)
echo  4. Back to previous menu
echo.
echo ======================================================================
set /p render_choice="Enter your choice (1-4): "

if "%render_choice%"=="1" (
    set enable_3d_rendering=--enable_3d_rendering
    set render_show_slice=
    goto SELECT_VIZ_CONFIG
)

if "%render_choice%"=="2" (
    set enable_3d_rendering=--enable_3d_rendering
    set render_show_slice=--render_show_slice
    goto SELECT_VIZ_CONFIG
)

if "%render_choice%"=="3" (
    set enable_3d_rendering=
    set render_show_slice=
    set viz_config=
    goto RUN_INFERENCE
)

if "%render_choice%"=="4" (
    if "%inference_mode%"=="single" goto SELECT_SAMPLE
    goto SELECT_NUM_VIS
)

echo Invalid choice! Press any key to try again...
pause >nul
goto SELECT_3D_RENDERING

:SELECT_VIZ_CONFIG
setlocal enabledelayedexpansion
cls
echo ======================================================================
echo Select Visualization Configuration
echo ======================================================================
echo.
echo Checkpoint:    %checkpoint%
echo Dataset:       %data_dir%
echo Split:         %split%
if "%inference_mode%"=="single" (
    echo Mode:          Single sample ^(%sample%^)
) else (
    echo Mode:          Batch ^(%num_vis% samples^)
)
echo 3D Rendering:  ENABLED
if "%render_show_slice%"=="--render_show_slice" (
    echo Slice Mode:    WITH slice visualization
) else (
    echo Slice Mode:    WITHOUT slice
)
echo.
echo Available visualization configurations:
echo.

REM Dynamically scan configs/inference/visualization directory
set idx=0
for %%f in (configs\inference\visualization\*.yaml) do (
    set /a idx+=1
    set "cfg!idx!=%%~nf"
    echo  !idx!. %%~nf
)
set config_count=!idx!

echo  0. No config ^(use command-line defaults^)
echo  99. Back to 3D rendering options
echo.
echo ======================================================================
set /p choice="Enter your choice (0-!config_count! or 99): "

if "!choice!"=="0" (
    endlocal
    set viz_config=
    goto RUN_INFERENCE
)

if "!choice!"=="99" (
    endlocal
    goto SELECT_3D_RENDERING
)

REM Validate choice
if !choice! gtr !config_count! (
    endlocal
    echo Invalid choice! Press any key to try again...
    pause >nul
    goto SELECT_VIZ_CONFIG
)

if !choice! lss 1 (
    endlocal
    echo Invalid choice! Press any key to try again...
    pause >nul
    goto SELECT_VIZ_CONFIG
)

REM Build config path
call set selected=%%cfg!choice!%%
endlocal & set viz_config=--viz-config configs/inference/visualization/%selected%.yaml
goto RUN_INFERENCE

:RUN_INFERENCE
cls
echo ======================================================================
echo Running Inference
echo ======================================================================
echo.
echo Checkpoint:    %checkpoint%
echo Dataset:       %data_dir%
echo Model Config:  %model_config%
echo Split:         %split%
if "%inference_mode%"=="single" (
    echo Mode:          Single sample
    echo Sample:        %sample%
) else (
    echo Mode:          Batch (all samples)
    echo Visualizations: %num_vis%
)
if "%enable_3d_rendering%"=="--enable_3d_rendering" (
    echo 3D Rendering:  ENABLED
    if not "%viz_config%"=="" (
        echo Viz Config:    %viz_config:--viz-config =%
    ) else (
        echo Viz Config:    Command-line defaults
    )
) else (
    echo 3D Rendering:  DISABLED
)
echo.

REM Extract checkpoint name and model (experiment) name from path
REM Example: outputs\training\confocal_128_v1_1_8_p4_r4\checkpoint_best.pth
REM -> ckpt_name: checkpoint_best
REM -> model_name: confocal_128_v1_1_8_p4_r4

REM Get just the filename without extension
for %%F in ("%checkpoint%") do (
    set "ckpt_name=%%~nF"
)

REM Get the parent directory name (experiment/model name)
for %%F in ("%checkpoint%") do (
    set "ckpt_parent=%%~dpF"
)
REM Remove trailing backslash then extract folder name
if "!ckpt_parent:~-1!"=="\" set "ckpt_parent=!ckpt_parent:~0,-1!"
for %%D in ("!ckpt_parent!") do set "model_name=%%~nxD"

REM Build output directory: outputs/inference/<model_name>_on_<dataset_name>/<checkpoint_name>/<split>/
set "output_dir=outputs\inference\!model_name!_on_%dataset_name%\%ckpt_name%\%split%"

echo Output directory: %output_dir%
echo.
pause

call conda activate raft-dvc-2
echo.

REM Build inference command based on mode
if "%inference_mode%"=="single" (
    REM Single sample mode
    if "%model_config%"=="" (
        python inference_test.py --checkpoint "%checkpoint%" --data_dir "%data_dir%" --split %split% --sample %sample% --output_dir "%output_dir%" %enable_3d_rendering% %viz_config% %render_show_slice%
    ) else (
        python inference_test.py --checkpoint "%checkpoint%" --model-config "%model_config%" --data_dir "%data_dir%" --split %split% --sample %sample% --output_dir "%output_dir%" %enable_3d_rendering% %viz_config% %render_show_slice%
    )
) else (
    REM Batch mode (all samples)
    if "%model_config%"=="" (
        python inference_test.py --checkpoint "%checkpoint%" --data_dir "%data_dir%" --split %split% --num_vis %num_vis% --output_dir "%output_dir%" %enable_3d_rendering% %viz_config% %render_show_slice%
    ) else (
        python inference_test.py --checkpoint "%checkpoint%" --model-config "%model_config%" --data_dir "%data_dir%" --split %split% --num_vis %num_vis% --output_dir "%output_dir%" %enable_3d_rendering% %viz_config% %render_show_slice%
    )
)

echo.
echo ======================================================================
echo Done! Check %output_dir% for results
echo ======================================================================
pause
goto INFERENCE_MENU

REM ======================================================================
REM DATASET GENERATION MENU
REM ======================================================================
:DATASET_MENU
cls
echo ======================================================================
echo                    Dataset Generation Menu
echo ======================================================================
echo.
echo  1. Generate Synthetic Dataset
echo      - Particle-based simulation
echo      - Fully configurable deformations
echo.
echo  2. Generate Dataset from Experimental TIF Images
echo      - Extract volumes from real TIF files
echo      - Apply deformations with known ground truth
echo.
echo  0. Back to main menu
echo.
echo ======================================================================
set /p dataset_type="Enter your choice (0-2): "

if "%dataset_type%"=="0" goto MAIN_MENU
if "%dataset_type%"=="1" goto DATASET_MENU_SYNTHETIC
if "%dataset_type%"=="2" goto DATASET_MENU_EXPERIMENTAL
echo Invalid choice! Press any key to try again...
pause >nul
goto DATASET_MENU

REM ======================================================================
REM SYNTHETIC DATASET GENERATION
REM ======================================================================
:DATASET_MENU_SYNTHETIC
cls
echo ======================================================================
echo                Synthetic Dataset Generation - Step 1/2
echo ======================================================================
echo.
echo Select Dataset Configuration
echo.
echo Available configurations in configs/data_generation/:
echo.

REM Collect all .yaml files in configs/data_generation/
set cfg_idx=0
for %%F in (configs\data_generation\*.yaml) do (
    set /a cfg_idx+=1
    set "cfg_file_!cfg_idx!=%%F"
    echo  !cfg_idx!. %%~nxF
)

echo.
echo  0. Back to dataset menu
echo.
echo ======================================================================
set /p cfg_choice="Select configuration (0-%cfg_idx%): "

if "%cfg_choice%"=="0" goto DATASET_MENU
if %cfg_choice% LEQ 0 goto DATASET_MENU_SYNTHETIC_INVALID
if %cfg_choice% GTR %cfg_idx% goto DATASET_MENU_SYNTHETIC_INVALID

REM Get selected config file
set selected_cfg=!cfg_file_%cfg_choice%!
echo.
echo Selected: %selected_cfg%
echo.
goto DATASET_MODE_SELECT

:DATASET_MENU_SYNTHETIC_INVALID
echo Invalid choice! Press any key to try again...
pause >nul
goto DATASET_MENU_SYNTHETIC

REM ======================================================================
REM EXPERIMENTAL DATASET GENERATION (FROM TIF)
REM ======================================================================
:DATASET_MENU_EXPERIMENTAL
cls
echo ======================================================================
echo          Experimental Dataset Generation - From TIF Images
echo ======================================================================
echo.
echo This tool generates datasets from real experimental TIF images:
echo   - Extracts 128^3 volumes from large TIF files
echo   - Applies controlled deformations
echo   - Creates ground truth flow fields
echo.
echo Prerequisites:
echo   1. Place raw TIF files in data_tif/experiment_name/
echo   2. Create config in configs/data_generation_from_experiments/
echo.
echo ======================================================================
echo.
echo Available configurations in configs/data_generation_from_experiments/:
echo.

REM Collect all .yaml files in configs/data_generation_from_experiments/
set exp_cfg_idx=0
for %%F in (configs\data_generation_from_experiments\*.yaml) do (
    set /a exp_cfg_idx+=1
    set "exp_cfg_file_!exp_cfg_idx!=%%F"
    echo  !exp_cfg_idx!. %%~nxF
)

if %exp_cfg_idx%==0 (
    echo  ^(No configurations found^)
    echo.
    echo Create a config file first! See example:
    echo   configs/data_generation_from_experiments/JinYang_confocal_beads_indentation.yaml
    echo.
    pause
    goto DATASET_MENU
)

echo.
echo  0. Back to dataset menu
echo.
echo ======================================================================
set /p exp_cfg_choice="Select configuration (0-%exp_cfg_idx%): "

if "%exp_cfg_choice%"=="0" goto DATASET_MENU
if %exp_cfg_choice% LEQ 0 goto DATASET_MENU_EXPERIMENTAL_INVALID
if %exp_cfg_choice% GTR %exp_cfg_idx% goto DATASET_MENU_EXPERIMENTAL_INVALID

REM Get selected config file
set selected_exp_cfg=!exp_cfg_file_%exp_cfg_choice%!

REM Check if raw data directory exists (read from config)
echo.
echo Checking configuration...
python -c "import yaml; cfg=yaml.safe_load(open('%selected_exp_cfg%')); print(cfg['input']['raw_data_dir'])" > temp_raw_dir.txt
set /p raw_dir=<temp_raw_dir.txt
del temp_raw_dir.txt

if not exist "%raw_dir%" (
    echo.
    echo ======================================================================
    echo ERROR: Raw data directory not found: %raw_dir%
    echo.
    echo Please check:
    echo   1. Config file: %selected_exp_cfg%
    echo   2. Raw data directory: %raw_dir%
    echo.
    echo Make sure your TIF files are in the correct location.
    echo ======================================================================
    pause
    goto DATASET_MENU_EXPERIMENTAL
)

REM Show summary
cls
echo ======================================================================
echo          Experimental Dataset Generation - Confirmation
echo ======================================================================
echo.
echo Configuration:  %selected_exp_cfg%
echo Raw data:       %raw_dir%
echo.
python -c "import yaml; cfg=yaml.safe_load(open('%selected_exp_cfg%')); print(f\"Dataset name:   {cfg['dataset_name']}\"); splits=cfg['splits']; total=splits['train']['num_samples']+splits['val']['num_samples']+splits['test']['num_samples']; print(f\"Total samples:  {total} (train={splits['train']['num_samples']}, val={splits['val']['num_samples']}, test={splits['test']['num_samples']})\"); print(f\"Output:         data/{cfg['dataset_name']}/\")"
echo.
echo This process will:
echo   1. Load TIF files from %raw_dir%
echo   2. Extract 128^3 volumes
echo   3. Apply deformations and create ground truth
echo   4. Save to data/^(dataset_name^)/
echo.
echo Estimated time: 10-30 minutes (depends on file size and config)
echo.
echo ======================================================================
set /p confirm="Start generation? (y/n): "

if /i not "%confirm%"=="y" goto DATASET_MENU_EXPERIMENTAL

REM Start generation
cls
echo ======================================================================
echo Generating Experimental Dataset
echo ======================================================================
echo.
call conda activate raft-dvc-2
echo.
python scripts/data_generation_from_experiments/generate_from_tif.py --config "%selected_exp_cfg%"
echo.
echo ======================================================================
echo Generation complete!
echo ======================================================================
pause
goto MAIN_MENU

:DATASET_MENU_EXPERIMENTAL_INVALID
echo Invalid choice! Press any key to try again...
pause >nul
goto DATASET_MENU_EXPERIMENTAL

:DATASET_MODE_SELECT
cls
echo ======================================================================
echo            Synthetic Dataset Generation - Step 2/2
echo ======================================================================
echo.
echo Selected config: %selected_cfg%
echo.
echo Choose generation mode:
echo.
echo  1. Full Dataset Generation
echo      - Generate all samples (train/val/test splits)
echo      - Estimated time: 30 min - 2 hours (depends on config)
echo.
echo  2. Preview Mode (Generate 1 Sample Only)
echo      - Generate single sample for quick preview
echo      - Check configuration before full generation
echo      - Output: Same directory with only 1 sample
echo.
echo  0. Back to config selection
echo.
echo ======================================================================
set /p mode_choice="Enter your choice (0-2): "

if "%mode_choice%"=="0" goto DATASET_MENU_SYNTHETIC
if "%mode_choice%"=="1" goto GEN_FULL
if "%mode_choice%"=="2" goto GEN_PREVIEW
echo Invalid choice! Press any key to try again...
pause >nul
goto DATASET_MODE_SELECT

:GEN_FULL
cls
echo ======================================================================
echo Generate Full Dataset
echo ======================================================================
echo.
echo Config: %selected_cfg%
echo.
echo This will generate all samples as specified in the config.
echo Check the config file for exact numbers (typically 800 train + 100 val + 100 test).
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause

call conda activate raft-dvc-2
echo.
echo Starting full dataset generation...
echo.
python scripts/data_generation/generate_confocal_dataset.py --config "%selected_cfg%"
echo.
echo ======================================================================
echo Generation complete!
echo ======================================================================
pause
goto MAIN_MENU

:GEN_PREVIEW
cls
echo ======================================================================
echo Preview Mode - Generate Single Sample
echo ======================================================================
echo.
echo Config: %selected_cfg%
echo.
echo Generating 1 sample for preview...
echo This allows you to check the configuration before full generation.
echo.
pause

call conda activate raft-dvc-2
echo.
echo Starting preview generation...
echo.

REM Create temporary preview config by modifying num_samples
REM Use Python to generate preview config
python -c "import yaml; import sys; cfg=yaml.safe_load(open('%selected_cfg%', encoding='utf-8')); cfg['dataset']['num_samples']={'train':1,'val':0,'test':0}; cfg['dataset']['output_dir']=cfg['dataset']['output_dir']+'_preview'; cfg['generation']['visualize_samples']=True; cfg['generation']['num_visualize']=1; yaml.dump(cfg, open('configs/data_generation/_preview_temp.yaml','w'), default_flow_style=False, allow_unicode=True)"

python scripts/data_generation/generate_confocal_dataset.py --config "configs/data_generation/_preview_temp.yaml"

REM Clean up temp config
del configs\data_generation\_preview_temp.yaml

echo.
echo ======================================================================
echo Preview generated! Check output directory with "_preview" suffix.
echo If satisfied, run full generation with mode 1.
echo ======================================================================
pause
goto MAIN_MENU

REM ======================================================================
REM TEST SET EVALUATION MENU
REM ======================================================================
:TESTSET_MENU
cls
echo ======================================================================
echo                    Test Set Evaluation Menu
echo ======================================================================
echo.
echo  1. Quick test (single test sample)
echo  2. Evaluate all test samples
echo  3. Evaluate with full visualization save
echo  4. Back to main menu
echo.
echo ======================================================================
set /p test_choice="Enter your choice (1-4): "

if "%test_choice%"=="1" goto TESTSET_SINGLE
if "%test_choice%"=="2" goto TESTSET_BATCH
if "%test_choice%"=="3" goto TESTSET_SAVEALL
if "%test_choice%"=="4" goto MAIN_MENU
echo Invalid choice! Press any key to try again...
pause >nul
goto TESTSET_MENU

:TESTSET_SINGLE
call conda activate raft-dvc-2
echo.
set /p test_idx="Enter test sample index (0-99, default=50): "
if "%test_idx%"=="" set test_idx=50
echo.
echo Testing sample %test_idx% from test set...
echo.
python inference_test.py --checkpoint outputs/training/confocal_baseline/checkpoint_best.pth --split test --sample %test_idx%
echo.
echo ======================================================================
echo Done! Check outputs/inference/
echo ======================================================================
pause
goto TESTSET_MENU

:TESTSET_BATCH
call conda activate raft-dvc-2
echo.
echo Evaluating entire test set (100 samples)...
echo This will compute metrics for all test samples (visualize first 3)
echo.
python inference_test.py --checkpoint outputs/training/confocal_baseline/checkpoint_best.pth --split test
echo.
echo ======================================================================
echo Done! Check outputs/inference/ for metrics and visualizations
echo ======================================================================
pause
goto TESTSET_MENU

:TESTSET_SAVEALL
cls
echo ======================================================================
echo Save All Test Set Visualizations
echo ======================================================================
echo.
echo WARNING: This will save visualizations for all 100 test samples
echo Expected size: ~200-300 MB
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto TESTSET_MENU

call conda activate raft-dvc-2
echo.
echo Generating visualizations for all test samples...
echo This may take 5-10 minutes...
echo.
python inference_test.py --checkpoint outputs/training/confocal_baseline/checkpoint_best.pth --split test --num_vis -1
echo.
echo ======================================================================
echo Done! All visualizations saved to outputs/inference/
echo ======================================================================
pause
goto TESTSET_MENU

REM ======================================================================
REM LEHU DATASET TESTING MENU
REM ======================================================================
:LEHU_MENU
cls
echo ======================================================================
echo                    Lehu Dataset Testing Menu
echo ======================================================================
echo.
echo  1. Test single sample
echo  2. Test all samples
echo  3. Inspect dataset structure
echo  4. Back to main menu
echo.
echo ======================================================================
set /p lehu_choice="Enter your choice (1-4): "

if "%lehu_choice%"=="1" goto LEHU_SINGLE
if "%lehu_choice%"=="2" goto LEHU_ALL
if "%lehu_choice%"=="3" goto LEHU_INSPECT
if "%lehu_choice%"=="4" goto MAIN_MENU
echo Invalid choice! Press any key to try again...
pause >nul
goto LEHU_MENU

:LEHU_SINGLE
call conda activate raft-dvc-2
echo.
set /p lehu_sample="Enter sample index (0-4): "
if "%lehu_sample%"=="" set lehu_sample=0
echo.
echo Testing Lehu sample %lehu_sample%...
echo.
python tools/debug/test_lehu_data.py --sample %lehu_sample%
echo.
echo ======================================================================
echo Done! Check outputs/lehu_inference/ for results
echo ======================================================================
pause
goto LEHU_MENU

:LEHU_ALL
call conda activate raft-dvc-2
echo.
echo Testing all Lehu samples...
echo.
python tools/debug/test_lehu_data.py --all
echo.
echo ======================================================================
echo Done! Check outputs/lehu_inference/ for results
echo ======================================================================
pause
goto LEHU_MENU

:LEHU_INSPECT
call conda activate raft-dvc-2
echo.
echo Inspecting Lehu dataset structure...
echo.
python tools/debug/inspect_lehu_data.py
echo.
echo ======================================================================
echo Dataset inspection complete!
echo ======================================================================
pause
goto LEHU_MENU

REM ======================================================================
REM EXIT
REM ======================================================================
:EXIT
cls
echo.
echo Thank you for using RAFT-DVC!
echo.
timeout /t 2 >nul
exit /b 0
