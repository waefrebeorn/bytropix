@echo off
cls
echo Activating virtual environment and running WuBuNestmRnaTrainer.py...
echo --- WuBu Nesting Nucleotide Model Training (Hybrid Spatial/Hyperbolic) ---

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
REM === Prepare Annotation Databases (using annotations_db.py)       ===
REM ====================================================================
set "DATA_DIR=data_h5"
REM --- Specify the *FILENAMES* for the GFF files ---
REM --- The script will try to download using default URLs if these files are missing ---
REM --- ADJUST THESE FILENAMES if needed ---
set "GENCODE_GFF_FILENAME=gencode.v46.annotation.gff3.gz"
set "REFSEQ_GFF_FILENAME=GRCh38_latest_genomic.gff.gz"

echo Checking/Downloading/Creating Annotation Databases in %DATA_DIR%...
REM --- Run the Python script to handle downloads and DB creation ---
REM --- You can add --skip_gencode or --skip_refseq flags here if desired ---
%PYTHON_EXE% annotations_db.py --output_dir "%DATA_DIR%" --gencode_gff_file "%GENCODE_GFF_FILENAME%" --refseq_gff_file "%REFSEQ_GFF_FILENAME%"

if errorlevel 1 (
    echo ERROR: Failed to prepare annotation databases. See messages above.
    echo Please check internet connection, disk space, URLs in annotations_db.py, and gffutils/requests/tqdm installation.
    pause
    exit /b 1
)
echo Annotation database step complete.

REM ====================================================================
REM === Set WuBu Nesting Nucleotide Model Parameters (Hybrid Spatial) ===
REM ====================================================================

REM --- IMPORTANT: Choose training mode ---
REM Option 1: Single dataset (DISABLED)
REM set "DATASET_NAME=rfam_seed"
REM set "USE_COMBINED_DATASET="
REM set "BALANCED_SAMPLING=--no-balanced_sampling"
REM set "MAX_COMBINED="

REM Option 2: Combined dataset (ENABLED)
set "DATASET_NAME=rfam_seed"
set "USE_COMBINED_DATASET=--use_combined_dataset"
set "BALANCED_SAMPLING=--balanced_sampling"
set "MAX_COMBINED=--max_combined_datasets 1"

REM --- Paths and Project Names ---
set "CHECKPOINT_DIR=C:\projects\bytropix\wubu_nest_nucleotide_checkpoints"
set "WANDB_PROJECT=WuBuNestingHybrid"
set "WANDB_ENTITY=waefrebeorn-wubu"

REM --- GFF Annotation DB Paths (Check existence after the script runs) ---
set "GENCODE_GFF_DB=%DATA_DIR%\gencode_annotations.db"
set "REFSEQ_GFF_DB=%DATA_DIR%\refseq_annotations.db"
set "GENCODE_GFF_FLAG="
if exist "%GENCODE_GFF_DB%" (
    set "GENCODE_GFF_FLAG=--gencode_annotation_db "%GENCODE_GFF_DB%""
    echo Found GENCODE DB: %GENCODE_GFF_DB%
) else (
    echo WARNING: GENCODE GFF DB not found after preparation script. Training may be affected.
)
set "REFSEQ_GFF_FLAG="
if exist "%REFSEQ_GFF_DB%" (
    set "REFSEQ_GFF_FLAG=--refseq_annotation_db "%REFSEQ_GFF_DB%""
    echo Found REFSEQ DB: %REFSEQ_GFF_DB%
) else (
    echo WARNING: REFSEQ GFF DB not found after preparation script. Training may be affected.
)


REM --- Sequence Model Base Config (Hybrid Spatial) ---
set "LOCAL_HIDDEN_SIZE=512"
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
REM Adjusted for 6GB VRAM - MANUAL SETTINGS
set "BATCH_SIZE=4"
set "GRAD_ACCUM_STEPS=16"
set "LEARNING_RATE=3e-4"
set "EPOCHS=15"
set "WEIGHT_DECAY=0.01"
set "MAX_GRAD_NORM=1.0"
set "MOMENTUM=0.9"

REM --- Optimizer Q-Learning Controller ---
REM Set Q_CONTROL_FLAG=--disable_q_learning to disable
set "Q_CONTROL_FLAG="

REM --- Loss Weights ---
set "STRUCTURE_LOSS_WEIGHT=0.2"
set "REGION_LOSS_WEIGHT=0.1"
set "PRED_STRUCT_WEIGHT=0.5"

REM --- Creative Batching (DISABLED for manual setting) ---
REM set "CREATIVE_BATCHING=--creative_batching"
REM set "VRAM_GB=6.0"
REM set "TARGET_EFFECTIVE_BATCH=64"
REM set "CREATIVE_BATCH_ARGS=--creative_batching_vram_gb %VRAM_GB% --target_effective_batch_size %TARGET_EFFECTIVE_BATCH% --creative_batching_safety_factor 2.0"
set "CREATIVE_BATCHING="
set "CREATIVE_BATCH_ARGS="

REM --- Logging & Saving ---
set "LOG_INTERVAL=50"
set "SAVE_INTERVAL=500"
set "LOG_LEVEL=INFO"

REM --- WandB ---
REM Set WANDB_CONTROL_FLAG=--disable_wandb to disable
set "WANDB_CONTROL_FLAG="

REM --- Misc ---
set "SEED=42"
set "NUM_WORKERS=1"
set "PREFETCH_FACTOR=2"
set "PIN_MEMORY=--pin_memory"

REM --- Control Flags ---
set "AMP_CONTROL_FLAG=--use_amp"
REM Use --detect_anomaly to enable
set "ANOMALY_CONTROL_FLAG="

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
REM Reduced max split size for 6GB
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64"
set "CUDA_LAUNCH_BLOCKING=0"
set "TOKENIZERS_PARALLELISM=false"

REM ======================================================
REM === Execute the Python Training Script (Single GPU) ===
REM ======================================================
echo Starting WuBu Nesting Nucleotide Model training (Hybrid Spatial)...
echo Current Directory: %CD%
echo Python Executable: %PYTHON_EXE%
echo Script Name: WuBuNestmRnaTrainer.py
echo Data Directory: %DATA_DIR%
echo Checkpoint Directory: %CHECKPOINT_DIR%
echo --- Key Training Params ---
echo Batch Size: %BATCH_SIZE%
echo Accum Steps: %GRAD_ACCUM_STEPS%
echo Max Grad Norm: %MAX_GRAD_NORM%
echo Learning Rate: %LEARNING_RATE%
echo Context Size: %CONTEXT_SIZE%
echo Hidden Size: %LOCAL_HIDDEN_SIZE%
echo ----------------------------
echo Dataset: %DATASET_NAME% (Combined: %USE_COMBINED_DATASET%)
echo Config: LR=%LEARNING_RATE%, GradNorm=%MAX_GRAD_NORM%, Batch=%BATCH_SIZE%, Accum=%GRAD_ACCUM_STEPS%, Epochs=%EPOCHS%
echo WuBu Config: Levels=%WUBU_LEVELS%, Dims=%WUBU_DIMS%
echo Creative Batching: DISABLED (Manual BS=%BATCH_SIZE%, Accum=%GRAD_ACCUM_STEPS%)
echo Architecture: Hidden=%LOCAL_HIDDEN_SIZE%, Enc-L=%NUM_ENCODER_LAYERS%, Dec-L=%NUM_DECODER_LAYERS%
echo Running with: %PYTHON_EXE% (single process)
echo.

REM --- Construct and run the command directly with python ---
%PYTHON_EXE% WuBuNestmRnaTrainer.py ^
 --context_size %CONTEXT_SIZE% ^
 --local_hidden_size %LOCAL_HIDDEN_SIZE% ^
 --num_encoder_layers %NUM_ENCODER_LAYERS% ^
 --num_decoder_layers %NUM_DECODER_LAYERS% ^
 --num_encoder_heads %NUM_ENCODER_HEADS% ^
 --num_decoder_heads %NUM_DECODER_HEADS% ^
 --encoder_max_len %ENCODER_MAX_LEN% ^
 --decoder_max_len %DECODER_MAX_LEN% ^
 --wubu_levels %WUBU_LEVELS% ^
 --wubu_dims %WUBU_DIMS% ^
 --batch_size %BATCH_SIZE% ^
 --grad_accum_steps %GRAD_ACCUM_STEPS% ^
 --learning_rate %LEARNING_RATE% ^
 --momentum %MOMENTUM% ^
 --epochs %EPOCHS% ^
 --weight_decay %WEIGHT_DECAY% ^
 --max_grad_norm %MAX_GRAD_NORM% ^
 --data_dir "%DATA_DIR%" ^
 --checkpoint_dir "%CHECKPOINT_DIR%" ^
 %USE_COMBINED_DATASET% ^
 %BALANCED_SAMPLING% ^
 %MAX_COMBINED% ^
 %GENCODE_GFF_FLAG% ^
 %REFSEQ_GFF_FLAG% ^
 --structure_loss_weight %STRUCTURE_LOSS_WEIGHT% ^
 --region_loss_weight %REGION_LOSS_WEIGHT% ^
 --predicted_structure_weight %PRED_STRUCT_WEIGHT% ^
 %Q_CONTROL_FLAG% ^
 %CREATIVE_BATCHING% ^
 %CREATIVE_BATCH_ARGS% ^
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
 %RESUME_FLAG%
pause
REM --- Check exit code ---
if errorlevel 1 (
    echo Script failed with error code %errorlevel%.
    echo.
    echo Attempting to fix potential HDF5 data issues...
    if exist "%DATA_DIR%\combined_rna_streams.h5" (
        del /f "%DATA_DIR%\combined_rna_streams.h5"
        echo Deleted combined HDF5 dataset: %DATA_DIR%\combined_rna_streams.h5
    )
    if exist "%DATA_DIR%\combined_rna_streams.h5.building" (
        del /f "%DATA_DIR%\combined_rna_streams.h5.building"
        echo Deleted partial combined HDF5 dataset: %DATA_DIR%\combined_rna_streams.h5.building
    )
    if exist "%DATA_DIR%\combined_rna_dataset_info_h5.json" (
        del /f "%DATA_DIR%\combined_rna_dataset_info_h5.json"
        echo Deleted combined HDF5 metadata: %DATA_DIR%\combined_rna_dataset_info_h5.json
    )
    REM Add cleanup for individual dataset HDF5 files if necessary
    REM Example: del /f "%DATA_DIR%\rfam_seed_streams.h5"
    echo.
    echo Cleanup attempted. Please try running the script again.
    echo If the error persists, consider:
    echo    1. Verifying Python dependencies (pytorch, h5py, triton, etc.).
    echo    2. Checking disk space and permissions in '%DATA_DIR%' and '%CHECKPOINT_DIR%'.
    echo    3. Reducing batch size or accumulation steps further.
    echo    4. Running with --detect_anomaly to pinpoint calculation errors (will be slow).
    echo    5. Ensuring WuBuNestmRnaTrainer.py is in the current directory (%CD%) or adjust PYTHON_EXE path.
) else (
    echo Script finished successfully.
)

echo.
echo Execution completed. Press any key to exit.
pause > nul
exit /b %errorlevel%
