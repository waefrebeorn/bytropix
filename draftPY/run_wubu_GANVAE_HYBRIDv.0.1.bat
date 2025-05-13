@echo OFF
REM ==================================================================================
REM Batch file for WuBuGAADHybridGen_v0.1.py - VAE-GAN Comprehensive Run
REM Goal: Enable all relevant features, including Optical Flow motion branch.
REM ==================================================================================
SETLOCAL ENABLEDELAYEDEXPANSION

REM --- Project and Script Paths ---
SET "PROJECT_ROOT=%~dp0.."
IF EXIST "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
) ELSE IF EXIST "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) ELSE (
    ECHO Python executable not found. Defaulting to "python".
    SET "PYTHON_EXE=python"
)
SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT_NAME=WuBuGAADHybridGen_v0.1.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

REM --- DDP Settings ---
SET "NPROC_PER_NODE=1"
SET "MASTER_ADDR=localhost"
SET "MASTER_PORT=29513"
REM Incremented port again

REM --- Data and Output Directories ---
SET "DATA_DIR_BASE=%PROJECT_ROOT%\data"
SET "CHECKPOINT_OUTPUT_DIR=%PROJECT_ROOT%\checkpoints\WuBuGAADHybridGen_v01_Run1_Comprehensive"
SET "VIDEO_DATA_PATH=%DATA_DIR_BASE%\demo_video_data_dir"
SET "VALIDATION_VIDEO_PATH="

REM --- Model & Training Core Hyperparameters ---
SET "IMAGE_H=176"
SET "IMAGE_W=320"
SET "NUM_CHANNELS=3"
SET "NUM_INPUT_FRAMES=10"
SET "NUM_PREDICT_FRAMES=2"
SET "FRAME_SKIP=1"
SET "LATENT_DIM=128"

REM --- GAAD Parameters (Appearance Regions) ---
SET "GAAD_NUM_REGIONS=24"
SET "GAAD_DECOMP_TYPE=hybrid"
SET "GAAD_MIN_SIZE_PX=5"

REM --- Encoder Parameters ---
SET "ENCODER_USE_ROI_ALIGN=true"
SET "ENCODER_PIXEL_PATCH_SIZE=16"
REM Only used if ENCODER_USE_ROI_ALIGN is false
SET "ENCODER_SHALLOW_CNN_CHANNELS=32"
SET "ENCODER_ROI_ALIGN_OUTPUT_H=4"
SET "ENCODER_ROI_ALIGN_OUTPUT_W=4"
SET "ENCODER_INITIAL_TANGENT_DIM=192"

REM --- Decoder/Generator Parameters ---
SET "DECODER_TYPE=patch_gen"
SET "DECODER_PATCH_GEN_SIZE=16"
SET "DECODER_PATCH_RESIZE_MODE=bilinear"

REM --- Discriminator Parameters ---
SET "DISCRIMINATOR_TYPE=regional_cnn"
REM Add specific args like --disc_cnn_channels if they exist in Python

REM --- WuBu Common Parameters ---
SET "WUBU_DROPOUT=0.1"

REM --- WuBu-S (Encoder Appearance) Parameters ---
SET "WUBU_S_NUM_LEVELS=3"
SET "WUBU_S_HYPERBOLIC_DIMS=96 64 48"
SET "WUBU_S_INITIAL_CURVATURES=1.0 0.7 0.5"
SET "WUBU_S_USE_ROTATION=false"
SET "WUBU_S_PHI_CURVATURE=true"
SET "WUBU_S_PHI_ROT_INIT=false"

REM --- WuBu-T (Encoder Temporal -> Latent) Parameters ---
SET "WUBU_T_NUM_LEVELS=3"
SET "WUBU_T_HYPERBOLIC_DIMS=192 128 96"
SET "WUBU_T_INITIAL_CURVATURES=1.0 0.7 0.5"
SET "WUBU_T_USE_ROTATION=false"
SET "WUBU_T_PHI_CURVATURE=true"
SET "WUBU_T_PHI_ROT_INIT=false"

REM --- Motion Branch Parameters ---
SET "USE_WUBU_MOTION_BRANCH=true"
SET "GAAD_MOTION_NUM_REGIONS=16"
SET "GAAD_MOTION_DECOMP_TYPE=hybrid"
SET "WUBU_M_NUM_LEVELS=3"
SET "WUBU_M_HYPERBOLIC_DIMS=128 96 64"
SET "WUBU_M_INITIAL_CURVATURES=1.0 0.8 0.6"
SET "WUBU_M_USE_ROTATION=false"
SET "WUBU_M_PHI_CURVATURE=true"
SET "WUBU_M_PHI_ROT_INIT=false"
SET "OPTICAL_FLOW_NET_TYPE=raft_small"
SET "FREEZE_FLOW_NET=true"
SET "FLOW_STATS_COMPONENTS=mag_mean angle_mean"

REM --- Training Parameters ---
SET "EPOCHS=5"
SET "GLOBAL_BATCH_SIZE=8"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
REM Python script has --grad_accum_steps
SET "LEARNING_RATE_GEN=2e-4"
SET "LEARNING_RATE_DISC=2e-4"
SET "RISGD_MAX_GRAD_NORM=5.0"
SET "GLOBAL_MAX_GRAD_NORM=6.0"
SET "TRAIN_LOG_INTERVAL=2"
SET "TRAIN_SAVE_INTERVAL=50000000"
REM Very large, effectively end of epoch / best only
SET "TRAIN_SEED=42"
SET "TRAIN_NUM_WORKERS=4"
SET "USE_AMP=false"
SET "DETECT_ANOMALY=false"
SET "LOG_GRAD_NORM=false"

REM --- VAE-GAN Loss Weights ---
SET "LAMBDA_RECON=1.0"
SET "LAMBDA_KL=0.01"
SET "LAMBDA_GAN=0.1"

REM --- Validation & Sampling ---
IF /I "%USE_LPIPS_FOR_VERIFICATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_lpips_for_verification"
SET "VALIDATION_SPLIT_FRACTION=0.1"
SET "VAL_BLOCK_SIZE=20" REM Added parameter for validation block size
SET "VAL_PRIMARY_METRIC=avg_val_recon_mse"
SET "NUM_VAL_SAMPLES_TO_LOG=2"
SET "DEMO_NUM_SAMPLES=4"

REM --- Control Flags ---
SET "Q_CONTROLLER_ENABLED=true"
SET "WANDB_ENABLED=true"
SET "WANDB_PROJECT=WuBuGAADHybridGenV01_Comprehensive"
SET "WANDB_RUN_NAME="
REM Optional: e.g., "Run1_LR2e-4_Batch12"

REM ==================================================================================
REM --- Script Execution Arguments Construction ---
SET "SCRIPT_ARGS="

REM --- Data and Output ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --video_data_path "%VIDEO_DATA_PATH%""
IF DEFINED VALIDATION_VIDEO_PATH (
    IF NOT "%VALIDATION_VIDEO_PATH%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_video_path "%VALIDATION_VIDEO_PATH%""
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR%""

REM --- Model Core ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --image_h %IMAGE_H%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --image_w %IMAGE_W%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_channels %NUM_CHANNELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_input_frames %NUM_INPUT_FRAMES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_predict_frames %NUM_PREDICT_FRAMES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --frame_skip %FRAME_SKIP%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --latent_dim %LATENT_DIM%"

REM --- GAAD (Appearance) ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_num_regions %GAAD_NUM_REGIONS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_decomposition_type %GAAD_DECOMP_TYPE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_min_size_px %GAAD_MIN_SIZE_PX%"

REM --- Encoder ---
IF /I "%ENCODER_USE_ROI_ALIGN%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_use_roi_align"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_shallow_cnn_channels %ENCODER_SHALLOW_CNN_CHANNELS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_roi_align_output_h %ENCODER_ROI_ALIGN_OUTPUT_H%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_roi_align_output_w %ENCODER_ROI_ALIGN_OUTPUT_W%"
) ELSE (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_pixel_patch_size %ENCODER_PIXEL_PATCH_SIZE%"
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --encoder_initial_tangent_dim %ENCODER_INITIAL_TANGENT_DIM%"

REM --- Decoder/Generator ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --decoder_type %DECODER_TYPE%"
IF /I "%DECODER_TYPE%"=="patch_gen" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --decoder_patch_gen_size %DECODER_PATCH_GEN_SIZE%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --decoder_patch_resize_mode %DECODER_PATCH_RESIZE_MODE%"
)

REM --- Discriminator ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --discriminator_type %DISCRIMINATOR_TYPE%"
REM Add --disc_cnn_channels if Python script accepts it, e.g.
REM SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_cnn_channels 32 64 128"

REM --- WuBu Common ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_dropout %WUBU_DROPOUT%"

REM --- WuBu-S ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_num_levels %WUBU_S_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_hyperbolic_dims %WUBU_S_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_initial_curvatures %WUBU_S_INITIAL_CURVATURES%"
IF /I "%WUBU_S_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_use_rotation"
IF /I "%WUBU_S_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_curvature"
IF /I "%WUBU_S_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_rotation_init"

REM --- WuBu-T ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_num_levels %WUBU_T_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_hyperbolic_dims %WUBU_T_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_initial_curvatures %WUBU_T_INITIAL_CURVATURES%"
IF /I "%WUBU_T_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_use_rotation"
IF /I "%WUBU_T_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_phi_influence_curvature"
IF /I "%WUBU_T_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_phi_influence_rotation_init"

REM --- Motion Branch ---
IF /I "%USE_WUBU_MOTION_BRANCH%"=="true" (
    ECHO Motion Branch IS ENABLED - Adding motion specific arguments
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_wubu_motion_branch"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gaad_motion_num_regions %GAAD_MOTION_NUM_REGIONS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gaad_motion_decomposition_type %GAAD_MOTION_DECOMP_TYPE%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_num_levels %WUBU_M_NUM_LEVELS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_hyperbolic_dims %WUBU_M_HYPERBOLIC_DIMS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_initial_curvatures %WUBU_M_INITIAL_CURVATURES%"
    IF /I "%WUBU_M_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_use_rotation"
    IF /I "%WUBU_M_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_phi_influence_curvature"
    IF /I "%WUBU_M_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_phi_influence_rotation_init"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optical_flow_net_type %OPTICAL_FLOW_NET_TYPE%"
    IF /I "%FREEZE_FLOW_NET%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --freeze_flow_net"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --flow_stats_components %FLOW_STATS_COMPONENTS%"
) ELSE (
    ECHO Motion Branch IS DISABLED
)

REM --- Training Loop ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --epochs %EPOCHS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --batch_size %BATCH_SIZE_PER_GPU%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --grad_accum_steps %GRAD_ACCUM_STEPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --learning_rate_gen %LEARNING_RATE_GEN%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --learning_rate_disc %LEARNING_RATE_DISC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --risgd_max_grad_norm %RISGD_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --global_max_grad_norm %GLOBAL_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --log_interval %TRAIN_LOG_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --save_interval %TRAIN_SAVE_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --seed %TRAIN_SEED%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_workers %TRAIN_NUM_WORKERS%"
IF /I "%USE_AMP%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_amp"
IF /I "%DETECT_ANOMALY%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --detect_anomaly"
IF /I "%LOG_GRAD_NORM%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --log_grad_norm"

REM --- VAE-GAN Loss Weights ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_recon %LAMBDA_RECON%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_kl %LAMBDA_KL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_gan %LAMBDA_GAN%"

REM --- Validation & Sampling ---
IF /I "%USE_LPIPS_FOR_VERIFICATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_lpips_for_verification"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_split_fraction %VALIDATION_SPLIT_FRACTION%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_block_size %VAL_BLOCK_SIZE%" REM Added validation block size argument
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_primary_metric %VAL_PRIMARY_METRIC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_val_samples_to_log %NUM_VAL_SAMPLES_TO_LOG%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --demo_num_samples %DEMO_NUM_SAMPLES%"

REM --- Control Flags ---
IF /I "%Q_CONTROLLER_ENABLED%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled"
IF /I "%WANDB_ENABLED%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb --wandb_project %WANDB_PROJECT%"
    IF DEFINED WANDB_RUN_NAME (
        IF NOT "%WANDB_RUN_NAME%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_run_name "%WANDB_RUN_NAME%""
    )
)
REM ==================================================================================

REM --- Environment Activation & Execution ---
ECHO ======================================================
ECHO WuBuGAADHybridGen VAE-GAN - Comprehensive Run
ECHO Python: %PYTHON_EXE%
ECHO Script Name: %SCRIPT_NAME%
ECHO Checkpoints: %CHECKPOINT_OUTPUT_DIR%
ECHO Motion Branch: %USE_WUBU_MOTION_BRANCH% (Optical Flow: %OPTICAL_FLOW_NET_TYPE%)
ECHO AMP: %USE_AMP%
ECHO Q-Controller: %Q_CONTROLLER_ENABLED%
ECHO WANDB: %WANDB_ENABLED% (Project: %WANDB_PROJECT%)
ECHO Learning Rate (Gen/Disc): %LEARNING_RATE_GEN% / %LEARNING_RATE_DISC%
ECHO Batch Size (Global/PerGPU): %GLOBAL_BATCH_SIZE% / %BATCH_SIZE_PER_GPU%
ECHO Grad Accum Steps: %GRAD_ACCUM_STEPS%
ECHO Validation Block Size: %VAL_BLOCK_SIZE% REM Added for clarity
ECHO ======================================================
ECHO.

IF NOT EXIST "%PYTHON_EXE%" (
    ECHO ERROR: Python executable not found at %PYTHON_EXE%
    GOTO :End
)

SET "VENV_ACTIVATE_PATH="
FOR %%F IN ("%PYTHON_EXE%") DO SET "VENV_ACTIVATE_PATH=%%~dpFactivate.bat"
IF EXIST "%VENV_ACTIVATE_PATH%" (
    ECHO Activating venv: %VENV_ACTIVATE_PATH%
    CALL "%VENV_ACTIVATE_PATH%"
    IF ERRORLEVEL 1 (
        ECHO WARNING: Failed to activate venv, proceeding.
    )
) ELSE (
    ECHO Venv activate.bat not found. Assuming active or Python in PATH.
)
ECHO.

IF NOT EXIST "%CHECKPOINT_OUTPUT_DIR%" MKDIR "%CHECKPOINT_OUTPUT_DIR%"
IF NOT EXIST "%DATA_DIR_BASE%" MKDIR "%DATA_DIR_BASE%"
IF NOT EXIST "%VIDEO_DATA_PATH%" MKDIR "%VIDEO_DATA_PATH%"
ECHO.

ECHO Starting training script: %SCRIPT_NAME%
ECHO Script path: %FULL_SCRIPT_PATH%
ECHO.
ECHO Full arguments being passed to Python:
ECHO "%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
ECHO.

REM PAUSE

IF %NPROC_PER_NODE% EQU 1 (
    ECHO Using direct Python execution...
    "%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
) ELSE (
    ECHO Using torchrun...
    REM Note: torch.distributed.run is preferred over torch.distributed.launch
    "%PYTHON_EXE%" -m torch.distributed.run --nproc_per_node=%NPROC_PER_NODE% --master_addr=%MASTER_ADDR% --master_port=%MASTER_PORT% --standalone --nnodes=1 "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
)

SET "EXIT_CODE=%ERRORLEVEL%"
ECHO.
IF %EXIT_CODE% NEQ 0 (
    ECHO * SCRIPT FAILED with exit code %EXIT_CODE% *
) ELSE (
    ECHO * SCRIPT FINISHED successfully *
)

:End
ECHO Batch script execution finished.
IF DEFINED PROMPT_AFTER_RUN ( PAUSE ) ELSE ( TIMEOUT /T 10 /NOBREAK >nul )
ENDLOCAL