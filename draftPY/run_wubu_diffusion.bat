@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM ============================================================================
REM == Batch Runner for WuBuDiffusion v4 (HGA-UNet)
REM ==
REM == This script is designed accordingto the "Batch File Design Document".
REM == It uses subroutines for modularity, validates paths, and provides
REM == clear configuration for the new HGA-UNet architecture.
REM ============================================================================

REM --- Main Execution Flow ---
CALL :Configure
CALL :FindLatestCheckpoint
CALL :AssembleArguments
CALL :PreRunChecks

REM --- Execute the Python Script ---
ECHO Starting HGA-UNet training script: !SCRIPT_NAME!
ECHO.
"!PYTHON_EXE!" "!FULL_SCRIPT_PATH!" !SCRIPT_ARGS!

CALL :PostRunChecks
GOTO :EOF

REM ============================================================================
REM == SUBROUTINES
REM ============================================================================

:Configure
    REM --- Core Paths and Script Configuration ---
    SET "PROJECT_ROOT=%~dp0.."
    IF EXIST "%PROJECT_ROOT%\venv\Scripts\python.exe" (
        SET "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
    ) ELSE IF EXIST "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
        SET "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
    ) ELSE (
        SET "PYTHON_EXE=python"
    )
    SET "SCRIPT_DIR=%~dp0"
    SET "SCRIPT_NAME=wubu_diffusion.py"
    SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"
    SET "CHECKPOINT_OUTPUT_DIR=%PROJECT_ROOT%\checkpoints\WuBuDiffusion_HGA_v4"
    SET "VIDEO_DATA_PATH=%PROJECT_ROOT%\data\vid_data"
    SET "LOAD_CHECKPOINT="

    REM --- Model and Data Parameters ---
    SET "IMAGE_W=256"
    SET "IMAGE_H=256"
    SET "NUM_CHANNELS=3"
    SET "NUM_WORKERS=0"

    REM --- HGA-UNet Architecture Parameters ---
    SET "HGA_PATCH_SIZE=8"
    SET "HGA_DIM=256"
    SET "HGA_POS_DIM=2"
    SET "HGA_POINCARE_C=1.0"
    SET "HGA_N_HEADS=8"
    SET "HGA_UNET_DEPTH=2"
    SET "HGA_NUM_LAYERS_PER_BLOCK=2"
    SET "HGA_KNN_K=16"
    SET "HGA_LEARNABLE_GEOMETRY=false"
    SET "HGA_LEARNABLE_CURVATURE=false"

    REM --- Diffusion Process Parameters ---
    SET "DIFFUSION_TIMESTEPS=1000"
    SET "DIFFUSION_BETA_SCHEDULE=cosine"

    REM --- Training and Optimizer Parameters ---
    SET "EPOCHS=200"
    SET "BATCH_SIZE=16"
    SET "LEARNING_RATE=2e-4"
    SET "OPTIMIZER=adamw_qcontrolled" REM Options: adamw, adamw_qcontrolled
    
    REM --- Q-Controller Specific Parameters (only used if OPTIMIZER is adamw_qcontrolled) ---
    SET "Q_LR=0.02"
    SET "Q_EPSILON=0.15"

    REM --- Logging and Saving Parameters ---
    SET "LOG_INTERVAL=50"
    SET "SAVE_INTERVAL=10"
    SET "WANDB_ENABLED=true"
    SET "WANDB_PROJECT=HGA-Diffusion-v4"
    
    REM --- Validation Parameters ---
    SET "VAL_FRACTION=0.1"
    SET "VAL_BATCH_SIZE=16"
EXIT /B 0


:FindLatestCheckpoint
    SET "LATEST_CKPT="
    IF NOT EXIST "!CHECKPOINT_OUTPUT_DIR!" GOTO :EOF

    REM Prioritize the highest epoch number
    FOR /F "tokens=1,2 delims=ep.pt" %%A IN ('dir /b /o-n "!CHECKPOINT_OUTPUT_DIR!\hga_ep*.pt"') DO (
        SET "LATEST_CKPT=!CHECKPOINT_OUTPUT_DIR!\hga_ep%%B.pt"
    )

    IF DEFINED LATEST_CKPT (
        SET "LOAD_CHECKPOINT=!LATEST_CKPT!"
    )
EXIT /B 0


:AssembleArguments
    SET "SCRIPT_ARGS="
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --video_data_path "!VIDEO_DATA_PATH!""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --checkpoint_dir "!CHECKPOINT_OUTPUT_DIR!""
    IF DEFINED LOAD_CHECKPOINT (
        IF NOT "!LOAD_CHECKPOINT!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_checkpoint "!LOAD_CHECKPOINT!""
    )

    REM --- Model and Data ---
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --image_h !IMAGE_H!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --image_w !IMAGE_W!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --num_channels !NUM_CHANNELS!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --num_workers !NUM_WORKERS!"

    REM --- HGA-UNet Architecture ---
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_patch_size !HGA_PATCH_SIZE!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_dim !HGA_DIM!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_pos_dim !HGA_POS_DIM!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_poincare_c !HGA_POINCARE_C!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_n_heads !HGA_N_HEADS!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_unet_depth !HGA_UNET_DEPTH!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_num_layers_per_block !HGA_NUM_LAYERS_PER_BLOCK!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_knn_k !HGA_KNN_K!"
    IF /I "!HGA_LEARNABLE_GEOMETRY!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_learnable_geometry"
    IF /I "!HGA_LEARNABLE_CURVATURE!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hga_learnable_curvature"
    
    REM --- Diffusion ---
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --diffusion_timesteps !DIFFUSION_TIMESTEPS!"

    REM --- Training and Optimizer ---
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --epochs !EPOCHS!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --batch_size !BATCH_SIZE!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --learning_rate !LEARNING_RATE!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optimizer !OPTIMIZER!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_lr !Q_LR!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_epsilon !Q_EPSILON!"

    REM --- Logging and Saving ---
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --log_interval !LOG_INTERVAL!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --save_interval !SAVE_INTERVAL!"
    IF /I "!WANDB_ENABLED!"=="true" (
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb"
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_project !WANDB_PROJECT!"
    )

    REM --- Validation ---
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --val_fraction !VAL_FRACTION!"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --val_batch_size !VAL_BATCH_SIZE!"
EXIT /B 0


:PreRunChecks
    ECHO ======================================================
    ECHO WuBuDiffusion V4 - HGA-UNet Execution
    ECHO ======================================================
    ECHO Python: !PYTHON_EXE!
    ECHO Script: !SCRIPT_NAME!
    ECHO Checkpoints: !CHECKPOINT_OUTPUT_DIR!
    ECHO Video Data: !VIDEO_DATA_PATH!
    ECHO Image Size: !IMAGE_H!x!IMAGE_W!
    ECHO HGA Patch Size: !HGA_PATCH_SIZE!
    ECHO HGA Dimension: !HGA_DIM!
    ECHO HGA U-Net Depth: !HGA_UNET_DEPTH!
    ECHO Optimizer: !OPTIMIZER!
    IF /I "!OPTIMIZER!"=="adamw_qcontrolled" (
        ECHO   Q-Controller LR: !Q_LR!
        ECHO   Q-Controller Epsilon: !Q_EPSILON!
    )
    ECHO WANDB Logging: !WANDB_ENABLED!
    IF DEFINED LOAD_CHECKPOINT (
        IF NOT "!LOAD_CHECKPOINT!"=="" ECHO Resuming from: !LOAD_CHECKPOINT!
    ) ELSE (
        ECHO Starting a new run.
    )
    ECHO ======================================================
    ECHO.

    IF NOT EXIST "!PYTHON_EXE%" (
        ECHO ERROR: Python executable not found at "!PYTHON_EXE!"
        EXIT /B 1
    )
    IF NOT EXIST "!CHECKPOINT_OUTPUT_DIR!" MKDIR "!CHECKPOINT_OUTPUT_DIR!"
    IF NOT EXIST "!VIDEO_DATA_PATH!" MKDIR "!VIDEO_DATA_PATH!"
EXIT /B 0


:PostRunChecks
    SET "EXIT_CODE=%ERRORLEVEL%"
    ECHO.
    ECHO ======================================================
    IF %EXIT_CODE% NEQ 0 (
        ECHO SCRIPT FAILED with exit code %EXIT_CODE%
    ) ELSE (
        ECHO SCRIPT FINISHED successfully
    )
    ECHO ======================================================
    REM Pause at the end to see output
    TIMEOUT /T 10 /NOBREAK >nul
EXIT /B %EXIT_CODE%
