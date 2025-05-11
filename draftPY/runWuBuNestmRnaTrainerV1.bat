@echo off
cls
echo Activating virtual environment and running wubu_nucleotide_trainer.py...
echo --- WuBu Nesting Nucleotide Model Training ---

REM --- Configuration ---
REM Set NUM_GPUS=1 for single GPU execution with python directly
set "NUM_GPUS=1"
set PYTHON_EXE=python

REM --- Check for Virtual Environment ---
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment activation script not found in .\venv\Scripts\
    echo Please ensure the virtual environment exists and is named 'venv'.
    pause
    exit /b 1
)

REM --- Activate Virtual Environment ---
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment. Check path and script integrity.
    pause
    exit /b 1
)
echo Virtual environment activated.
REM Update PYTHON_EXE to point to the venv python after activation
set PYTHON_EXE=venv\Scripts\python.exe

REM ====================================================================
REM === Set WuBu Nesting Nucleotide Model Parameters                 ===
REM ====================================================================

REM --- IMPORTANT: Choose training mode ---
REM Option 1: Single dataset (disabled)
REM set "DATASET_NAME=rfam_seed"
REM set "USE_COMBINED_DATASET="
REM set "BALANCED_SAMPLING="
REM set "MAX_COMBINED="

REM Option 2: Combined dataset (ENABLED)
set "DATASET_NAME=rfam_seed"
set "USE_COMBINED_DATASET=--use_combined_dataset"
set "BALANCED_SAMPLING=--balanced_sampling"
set "MAX_COMBINED=--max_combined_datasets 1"

REM --- Paths and Project Names ---
set "DATA_DIR=data"
set "CHECKPOINT_DIR=C:\projects\bytropix\wubu_nest_nucleotide_checkpoints"
set "WANDB_PROJECT=WuBuNestingNucleotide"
set "WANDB_ENTITY=waefrebeorn-wubu"

REM --- Sequence Model Base Config ---
set "LOCAL_HIDDEN_SIZE=384"
set "CONTEXT_SIZE=256"
set "ENCODER_MAX_LEN=1024"
set "DECODER_MAX_LEN=2048"
set "NUM_ENCODER_LAYERS=6"
set "NUM_ENCODER_HEADS=8"
set "NUM_DECODER_LAYERS=8"
set "NUM_DECODER_HEADS=8"

REM --- WuBu Nesting Config ---
set "WUBU_LEVELS=4"
set "WUBU_DIMS=128 96 64 32"

REM --- Training Hyperparameters ---
set "BATCH_SIZE=32"
set "GRAD_ACCUM_STEPS=16"
set "LEARNING_RATE=3e-4"
set "EPOCHS=15"
set "WEIGHT_DECAY=0.01"
set "MAX_GRAD_NORM=9001.0"
set "MOMENTUM=0.9"

REM --- Optimizer Q-Learning Controller ---
REM Set Q_CONTROL_FLAG=--disable_q_learning to disable
set "Q_CONTROL_FLAG="

REM --- Creative Batching ---
REM Note: Setting a more conservative target batch size for combined dataset
REM set "CREATIVE_BATCHING=--creative_batching"
set "CREATIVE_BATCHING="
set "VRAM_GB=5.5"
REM Reduced target effective batch size slightly given VRAM constraints
set "TARGET_EFFECTIVE_BATCH=128"
set "CREATIVE_BATCH_ARGS=--creative_batching_vram_gb %VRAM_GB% --target_effective_batch_size %TARGET_EFFECTIVE_BATCH% --creative_batching_safety_factor 2.0"

REM --- Logging & Saving ---
set "LOG_INTERVAL=1"
set "SAVE_INTERVAL=0"
set "LOG_LEVEL=INFO"

REM --- WandB ---
REM Set WANDB_CONTROL_FLAG=--disable_wandb to disable
set "WANDB_CONTROL_FLAG="

REM --- Misc ---
set "SEED=42"
set "NUM_WORKERS=2"
set "PREFETCH_FACTOR=2"
set "PIN_MEMORY=--pin_memory"

REM --- Control Flags ---
set "AMP_CONTROL_FLAG=--use_amp"
set "ANOMALY_CONTROL_FLAG=" REM Use --detect_anomaly to enable

REM --- NumPy Pickle Flag ---
set "NUMPY_ALLOW_PICKLE_FLAG=--numpy_allow_pickle"

REM --- Resume Flag ---
setlocal EnableDelayedExpansion
set "RECENT_CHECKPOINT="
if exist "%CHECKPOINT_DIR%" (
    REM Correctly escape paths with spaces for the dir command
    for /f "delims=" %%i in ('dir /b /o-d "%CHECKPOINT_DIR%\checkpoint_*.pt" 2^>nul') do (
        if "!RECENT_CHECKPOINT!"=="" set "RECENT_CHECKPOINT=%%i"
        goto :found_checkpoint
    )
)
:found_checkpoint

REM Ask to resume from most recent checkpoint
if not "!RECENT_CHECKPOINT!"=="" (
    echo Most recent checkpoint found: !RECENT_CHECKPOINT!
    set /p USE_CHECKPOINT="Resume from this checkpoint? (Y/N): "
    if /i "!USE_CHECKPOINT!"=="Y" (
        REM Ensure the path in the flag is properly quoted
        set "RESUME_FLAG=--load_checkpoint "%CHECKPOINT_DIR%\!RECENT_CHECKPOINT!""
        echo Will resume from: %CHECKPOINT_DIR%\!RECENT_CHECKPOINT!
    ) else (
        set "RESUME_FLAG="
        echo Starting fresh training run.
    )
) else (
    echo No existing checkpoints found. Starting fresh training run.
    set "RESUME_FLAG="
)
endlocal & set "RESUME_FLAG=%RESUME_FLAG%"

REM --- System Optimizations ---
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" REM Reduced max split size
set "CUDA_LAUNCH_BLOCKING=0"
set "TOKENIZERS_PARALLELISM=false"

REM ======================================================
REM === Execute the Python Training Script (Single GPU) ===
REM ======================================================
echo Starting WuBu Nesting Nucleotide Model training...
echo Dataset: %DATASET_NAME% (Combined: %USE_COMBINED_DATASET%)
echo Config: LR=%LEARNING_RATE%, GradNorm=%MAX_GRAD_NORM%, Batch=%BATCH_SIZE%, Accum=%GRAD_ACCUM_STEPS%, Epochs=%EPOCHS%
echo WuBu Config: Levels=%WUBU_LEVELS%, Dims=%WUBU_DIMS%
echo Creative Batching: VRAM=%VRAM_GB%GB, Target BS=%TARGET_EFFECTIVE_BATCH%
echo Architecture: Hidden=%LOCAL_HIDDEN_SIZE%, Enc-Layers=%NUM_ENCODER_LAYERS%, Dec-Layers=%NUM_DECODER_LAYERS%
echo Running with: %PYTHON_EXE% (single process)
echo.

REM --- Construct and run the command directly with python ---
%PYTHON_EXE% WuBuNestmRnaTrainerV1.py ^
  --dataset_name "%DATASET_NAME%" ^
  %USE_COMBINED_DATASET% ^
  %BALANCED_SAMPLING% ^
  %MAX_COMBINED% ^
  %CREATIVE_BATCHING% ^
  %CREATIVE_BATCH_ARGS% ^
  --data_dir "%DATA_DIR%" ^
  --checkpoint_dir "%CHECKPOINT_DIR%" ^
  --local_hidden_size %LOCAL_HIDDEN_SIZE% ^
  --context_size %CONTEXT_SIZE% ^
  --encoder_max_len %ENCODER_MAX_LEN% ^
  --decoder_max_len %DECODER_MAX_LEN% ^
  --num_encoder_layers %NUM_ENCODER_LAYERS% ^
  --num_encoder_heads %NUM_ENCODER_HEADS% ^
  --num_decoder_layers %NUM_DECODER_LAYERS% ^
  --num_decoder_heads %NUM_DECODER_HEADS% ^
  --wubu_levels %WUBU_LEVELS% ^
  --wubu_dims %WUBU_DIMS% ^
  --batch_size %BATCH_SIZE% ^
  --grad_accum_steps %GRAD_ACCUM_STEPS% ^
  --learning_rate %LEARNING_RATE% ^
  --momentum %MOMENTUM% ^
  --epochs %EPOCHS% ^
  --weight_decay %WEIGHT_DECAY% ^
  --max_grad_norm %MAX_GRAD_NORM% ^
  %Q_CONTROL_FLAG% ^
  --log_interval %LOG_INTERVAL% ^
  --save_interval %SAVE_INTERVAL% ^
  --log_level %LOG_LEVEL% ^
  %WANDB_CONTROL_FLAG% ^
  --wandb_project "%WANDB_PROJECT%" ^
  --wandb_entity "%WANDB_ENTITY%" ^
  --seed %SEED% ^
  --num_workers %NUM_WORKERS% ^
  --prefetch_factor %PREFETCH_FACTOR% ^
  %PIN_MEMORY% ^
  %AMP_CONTROL_FLAG% ^
  %ANOMALY_CONTROL_FLAG% ^
  %NUMPY_ALLOW_PICKLE_FLAG% ^
  %RESUME_FLAG%
pause
REM --- Check exit code ---
if errorlevel 1 (
    echo Script failed with error code %errorlevel%.
    echo.
    echo Attempting to fix the issue - deleting potentially corrupted combined dataset files...
    if exist "%DATA_DIR%\combined_rna_indices.npy" (
        del /f "%DATA_DIR%\combined_rna_indices.npy"
        echo Deleted combined dataset file: %DATA_DIR%\combined_rna_indices.npy
    )
    if exist "%DATA_DIR%\combined_rna_indices.npy.part" (
        del /f "%DATA_DIR%\combined_rna_indices.npy.part"
        echo Deleted partial combined dataset file: %DATA_DIR%\combined_rna_indices.npy.part
    )
    if exist "%DATA_DIR%\combined_rna_indices.npy.part_resize" (
        del /f "%DATA_DIR%\combined_rna_indices.npy.part_resize"
        echo Deleted resize temporary file: %DATA_DIR%\combined_rna_indices.npy.part_resize
    )
    if exist "%DATA_DIR%\combined_rna_dataset_info.json" (
        del /f "%DATA_DIR%\combined_rna_dataset_info.json"
        echo Deleted combined dataset metadata: %DATA_DIR%\combined_rna_dataset_info.json
    )
    echo.
    echo Cleanup attempted. Please try running the script again.
    echo If the error persists, consider:
    echo   1. Running with a single dataset (Option 1 at the top of this script).
    echo   2. Checking disk space and permissions in the '%DATA_DIR%' directory.
    echo   3. Ensuring NumPy and other dependencies are correctly installed.
) else (
    echo Script finished successfully.
)

echo.
echo Execution completed. Press any key to exit.
pause > nul
exit /b %errorlevel%