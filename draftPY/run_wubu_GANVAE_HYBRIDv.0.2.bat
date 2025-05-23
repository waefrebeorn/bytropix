@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM =====================================================================
REM Project and Python Setup
REM =====================================================================
SET "PROJECT_ROOT=%~dp0.."
IF EXIST "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
) ELSE IF EXIST "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) ELSE (
    SET "PYTHON_EXE=python"
)
SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT_NAME=WuBuGAADHybridGen_v0.2.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

REM =====================================================================
REM DDP Configuration
REM =====================================================================
SET "NPROC_PER_NODE=1"
SET "MASTER_ADDR=localhost"
SET "MASTER_PORT=29515"

REM =====================================================================
REM Path Configuration
REM =====================================================================
SET "DATA_DIR_BASE=%PROJECT_ROOT%\data"
SET "CHECKPOINT_OUTPUT_DIR=%PROJECT_ROOT%\checkpoints\WuBuGAADHybridGen_v02_DFT_Run_Latest"
SET "VIDEO_DATA_PATH=%DATA_DIR_BASE%\demo_video_data_dir_dft"
SET "VALIDATION_VIDEO_PATH="
SET "LOAD_CHECKPOINT="

REM =====================================================================
REM Data and Model Core Configuration
REM =====================================================================
SET "IMAGE_H=256"
SET "IMAGE_W=256"
SET "NUM_CHANNELS=3"
SET "NUM_INPUT_FRAMES=10"
SET "NUM_PREDICT_FRAMES=3"
SET "FRAME_SKIP=1"
SET "LATENT_DIM=512"

REM =====================================================================
REM GAAD Configuration
REM =====================================================================
SET "GAAD_NUM_REGIONS=24"
SET "GAAD_DECOMP_TYPE=hybrid"
SET "GAAD_MIN_SIZE_PX=8"

REM =====================================================================
REM DFT Configuration (NEW for v0.2)
REM =====================================================================
SET "USE_DFT_FEATURES_APPEARANCE=true"
SET "DFT_PATCH_SIZE_H=16"
SET "DFT_PATCH_SIZE_W=16"
SET "DFT_NORM_SCALE_VIDEO=20.0"

REM =====================================================================
REM Encoder Architecture Configuration
REM =====================================================================
SET "ENCODER_USE_ROI_ALIGN=true"
SET "ENCODER_PIXEL_PATCH_SIZE=16"
SET "ENCODER_SHALLOW_CNN_CHANNELS=32"
SET "ENCODER_ROI_ALIGN_OUTPUT_H=4"
SET "ENCODER_ROI_ALIGN_OUTPUT_W=4"
SET "ENCODER_INITIAL_TANGENT_DIM=192"

REM =====================================================================
REM Generator Architecture Configuration
REM =====================================================================
SET "GEN_TEMPORAL_KERNEL_SIZE=3"
SET "GEN_FINAL_CONV_KERNEL_SPATIAL=3"
SET "GEN_USE_GAAD_FILM_CONDITION=true"

REM =====================================================================
REM Discriminator Architecture Configuration
REM =====================================================================
SET "DISCRIMINATOR_TYPE=spatio_temporal_cnn"
SET "DISC_APPLY_SPECTRAL_NORM=true"
SET "DISC_BASE_DISC_CHANNELS=64"
SET "DISC_MAX_DISC_CHANNELS=512"
SET "DISC_TEMPORAL_KERNEL_SIZE=3"
SET "DISC_MIN_HIDDEN_FC_DIM=128"
SET "DISC_MAX_HIDDEN_FC_DIM=512"
SET "DISC_USE_GAAD_FILM_CONDITION=true"
SET "DISC_GAAD_CONDITION_DIM_DISC=64"
SET "DISC_PATCH_SIZE=16"
SET "DISC_CNN_CHANNELS_2D=64 128 256"

REM =====================================================================
REM WuBu Configuration (Common)
REM =====================================================================
SET "WUBU_DROPOUT=0.1"

REM =====================================================================
REM WuBu-S (Appearance) Configuration
REM =====================================================================
SET "WUBU_S_NUM_LEVELS=2"
SET "WUBU_S_HYPERBOLIC_DIMS=128 64"
SET "WUBU_S_INITIAL_CURVATURES=1.0 0.7"
SET "WUBU_S_USE_ROTATION=false"
SET "WUBU_S_PHI_CURVATURE=true"
SET "WUBU_S_PHI_ROT_INIT=false"

REM =====================================================================
REM WuBu-T (Temporal Aggregation) Configuration
REM =====================================================================
SET "WUBU_T_NUM_LEVELS=2"
SET "WUBU_T_HYPERBOLIC_DIMS=256 128"
SET "WUBU_T_INITIAL_CURVATURES=1.0 0.7"
SET "WUBU_T_USE_ROTATION=false"
SET "WUBU_T_PHI_CURVATURE=true"
SET "WUBU_T_PHI_ROT_INIT=false"

REM =====================================================================
REM WuBu-M (Motion) Configuration
REM =====================================================================
SET "USE_WUBU_MOTION_BRANCH=true"
SET "GAAD_MOTION_NUM_REGIONS=16"
SET "GAAD_MOTION_DECOMP_TYPE=hybrid"
SET "WUBU_M_NUM_LEVELS=2"
SET "WUBU_M_HYPERBOLIC_DIMS=128 64"
SET "WUBU_M_INITIAL_CURVATURES=1.0 0.8"
SET "WUBU_M_USE_ROTATION=true"
SET "WUBU_M_PHI_CURVATURE=true"
SET "WUBU_M_PHI_ROT_INIT=true"
SET "OPTICAL_FLOW_NET_TYPE=raft_small"
SET "FREEZE_FLOW_NET=true"
SET "FLOW_STATS_COMPONENTS=mag_mean angle_mean"

REM =====================================================================
REM Training Hyperparameters
REM =====================================================================
SET "EPOCHS=2500"
SET "GLOBAL_BATCH_SIZE=1"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
SET "LEARNING_RATE_GEN=2e-4"
SET "LEARNING_RATE_DISC=1e-4"
SET "RISGD_MAX_GRAD_NORM=2.0"
SET "GLOBAL_MAX_GRAD_NORM=2.0"
SET "TRAIN_LOG_INTERVAL=2"
SET "TRAIN_SAVE_INTERVAL=1000000"
SET "TRAIN_SEED=42"
SET "TRAIN_NUM_WORKERS=4"
SET "USE_AMP=false"
SET "DETECT_ANOMALY=false"
SET "LOG_GRAD_NORM=false"
SET "LOAD_STRICT=true"

REM =====================================================================
REM Loss Weights
REM =====================================================================
SET "LAMBDA_RECON=10.0"
SET "LAMBDA_KL=0.0001"
SET "LAMBDA_GAN=1.0"

REM =====================================================================
REM Q-Controller for Lambda_KL (Scheduled)
REM =====================================================================
SET "LAMBDA_KL_UPDATE_INTERVAL=25"
SET "MIN_LAMBDA_KL_Q_CONTROL=1e-7"
SET "MAX_LAMBDA_KL_Q_CONTROL=0.1"

REM =====================================================================
REM Validation and Logging
REM =====================================================================
SET "USE_LPIPS_FOR_VERIFICATION=true"
SET "VALIDATION_SPLIT_FRACTION=0.1"
SET "VAL_BLOCK_SIZE=20"
SET "VAL_PRIMARY_METRIC=avg_val_psnr"
SET "NUM_VAL_SAMPLES_TO_LOG=2"
SET "DEMO_NUM_SAMPLES=4"

REM =====================================================================
REM Experiment Control
REM =====================================================================
SET "Q_CONTROLLER_ENABLED=true"
SET "WANDB_ENABLED=true"
SET "WANDB_PROJECT=WuBuGAADHybridGenV02_DFT"
SET "WANDB_RUN_NAME="
SET "WANDB_LOG_TRAIN_RECON_INTERVAL=25"
SET "WANDB_LOG_FIXED_NOISE_INTERVAL=100"

REM =====================================================================
REM Script Argument Assembly
REM =====================================================================
SET "SCRIPT_ARGS="

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --video_data_path "%VIDEO_DATA_PATH%""
IF DEFINED VALIDATION_VIDEO_PATH (
    IF NOT "%VALIDATION_VIDEO_PATH%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_video_path "%VALIDATION_VIDEO_PATH%""
)
IF DEFINED LOAD_CHECKPOINT (
    IF NOT "%LOAD_CHECKPOINT%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --load_checkpoint "%LOAD_CHECKPOINT%""
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR%""

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --image_h %IMAGE_H%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --image_w %IMAGE_W%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_channels %NUM_CHANNELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_input_frames %NUM_INPUT_FRAMES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_predict_frames %NUM_PREDICT_FRAMES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --frame_skip %FRAME_SKIP%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --latent_dim %LATENT_DIM%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_num_regions %GAAD_NUM_REGIONS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_decomposition_type %GAAD_DECOMP_TYPE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_min_size_px %GAAD_MIN_SIZE_PX%"

REM DFT Args
IF /I "%USE_DFT_FEATURES_APPEARANCE%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_dft_features_appearance"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dft_patch_size_h %DFT_PATCH_SIZE_H%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dft_patch_size_w %DFT_PATCH_SIZE_W%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dft_norm_scale_video %DFT_NORM_SCALE_VIDEO%"
)

REM Encoder Args
IF /I "%ENCODER_USE_ROI_ALIGN%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_use_roi_align"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_shallow_cnn_channels %ENCODER_SHALLOW_CNN_CHANNELS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_roi_align_output_h %ENCODER_ROI_ALIGN_OUTPUT_H%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_roi_align_output_w %ENCODER_ROI_ALIGN_OUTPUT_W%"
) ELSE (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_pixel_patch_size %ENCODER_PIXEL_PATCH_SIZE%"
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --encoder_initial_tangent_dim %ENCODER_INITIAL_TANGENT_DIM%"

REM Generator Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gen_temporal_kernel_size %GEN_TEMPORAL_KERNEL_SIZE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gen_final_conv_kernel_spatial %GEN_FINAL_CONV_KERNEL_SPATIAL%"
IF /I "%GEN_USE_GAAD_FILM_CONDITION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gen_use_gaad_film_condition"

REM Discriminator Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --discriminator_type %DISCRIMINATOR_TYPE%"
IF /I "%DISC_APPLY_SPECTRAL_NORM%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_apply_spectral_norm"

IF /I "%DISCRIMINATOR_TYPE%"=="spatio_temporal_cnn" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_base_disc_channels %DISC_BASE_DISC_CHANNELS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_max_disc_channels %DISC_MAX_DISC_CHANNELS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_temporal_kernel_size %DISC_TEMPORAL_KERNEL_SIZE%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_min_hidden_fc_dim %DISC_MIN_HIDDEN_FC_DIM%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_max_hidden_fc_dim %DISC_MAX_HIDDEN_FC_DIM%"
    IF /I "%DISC_USE_GAAD_FILM_CONDITION%"=="true" (
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_use_gaad_film_condition"
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_gaad_condition_dim_disc %DISC_GAAD_CONDITION_DIM_DISC%"
    )
) ELSE IF /I "%DISCRIMINATOR_TYPE%"=="regional_cnn" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_patch_size %DISC_PATCH_SIZE%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_cnn_channels_2d %DISC_CNN_CHANNELS_2D%"
)

REM WuBu Args (Common)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_dropout %WUBU_DROPOUT%"

REM WuBu-S Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_num_levels %WUBU_S_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_hyperbolic_dims %WUBU_S_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_initial_curvatures %WUBU_S_INITIAL_CURVATURES%"
IF /I "%WUBU_S_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_use_rotation"
IF /I "%WUBU_S_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_curvature"
IF /I "%WUBU_S_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_rotation_init"

REM WuBu-T Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_num_levels %WUBU_T_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_hyperbolic_dims %WUBU_T_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_initial_curvatures %WUBU_T_INITIAL_CURVATURES%"
IF /I "%WUBU_T_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_use_rotation"
IF /I "%WUBU_T_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_phi_influence_curvature"
IF /I "%WUBU_T_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_phi_influence_rotation_init"

REM WuBu-M Args
IF /I "%USE_WUBU_MOTION_BRANCH%"=="true" (
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
)

REM Training Args
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
IF /I "%LOAD_STRICT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_strict"

REM Loss Weight Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_recon %LAMBDA_RECON%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_kl %LAMBDA_KL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_gan %LAMBDA_GAN%"

REM Lambda KL Q-Control Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_kl_update_interval %LAMBDA_KL_UPDATE_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --min_lambda_kl_q_control %MIN_LAMBDA_KL_Q_CONTROL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --max_lambda_kl_q_control %MAX_LAMBDA_KL_Q_CONTROL%"

REM Validation Args
IF /I "%USE_LPIPS_FOR_VERIFICATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_lpips_for_verification"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_split_fraction %VALIDATION_SPLIT_FRACTION%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_block_size %VAL_BLOCK_SIZE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_primary_metric %VAL_PRIMARY_METRIC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_val_samples_to_log %NUM_VAL_SAMPLES_TO_LOG%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --demo_num_samples %DEMO_NUM_SAMPLES%"

REM Experiment Control Args
IF /I "%Q_CONTROLLER_ENABLED%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled"
IF /I "%WANDB_ENABLED%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb --wandb_project %WANDB_PROJECT%"
    IF DEFINED WANDB_RUN_NAME (
        IF NOT "%WANDB_RUN_NAME%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_run_name "%WANDB_RUN_NAME%""
    )
)
IF %WANDB_LOG_TRAIN_RECON_INTERVAL% GTR 0 SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_log_train_recon_interval %WANDB_LOG_TRAIN_RECON_INTERVAL%"
IF %WANDB_LOG_FIXED_NOISE_INTERVAL% GTR 0 SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_log_fixed_noise_samples_interval %WANDB_LOG_FIXED_NOISE_INTERVAL%"


REM =====================================================================
REM Pre-Run Echo
REM =====================================================================
ECHO ======================================================
ECHO WuBuGAADHybridGen VAE-GAN DFT - Comprehensive Run
ECHO Python: %PYTHON_EXE%
ECHO Script Name: %SCRIPT_NAME%
ECHO Checkpoints: %CHECKPOINT_OUTPUT_DIR%
ECHO DFT Appearance: %USE_DFT_FEATURES_APPEARANCE%
ECHO Motion Branch: %USE_WUBU_MOTION_BRANCH% (Optical Flow: %OPTICAL_FLOW_NET_TYPE%)
ECHO Generator FiLM: %GEN_USE_GAAD_FILM_CONDITION%
ECHO Discriminator Type: %DISCRIMINATOR_TYPE%
ECHO Discriminator GAAD FiLM: %DISC_USE_GAAD_FILM_CONDITION% (Note: Only if spatio_temporal_cnn)
ECHO Discriminator Spectral Norm: %DISC_APPLY_SPECTRAL_NORM%
ECHO AMP: %USE_AMP%
ECHO Q-Controller: %Q_CONTROLLER_ENABLED% (Lambda_KL Update Interval: %LAMBDA_KL_UPDATE_INTERVAL%)
ECHO WANDB: %WANDB_ENABLED% (Project: %WANDB_PROJECT%)
ECHO Learning Rate (Gen/Disc): %LEARNING_RATE_GEN% / %LEARNING_RATE_DISC%
ECHO Batch Size (Global/PerGPU): %GLOBAL_BATCH_SIZE% / %BATCH_SIZE_PER_GPU%
ECHO Grad Accum Steps: %GRAD_ACCUM_STEPS%
ECHO Validation Block Size: %VAL_BLOCK_SIZE%
ECHO ======================================================
ECHO.

REM =====================================================================
REM Environment Setup and Execution
REM =====================================================================
IF NOT EXIST "%PYTHON_EXE%" (
    ECHO ERROR: Python executable not found at %PYTHON_EXE%
    GOTO :End
)

SET "VENV_ACTIVATE_PATH="
FOR %%F IN ("%PYTHON_EXE%") DO SET "VENV_ACTIVATE_PATH=%%~dpFactivate.bat"
IF EXIST "%VENV_ACTIVATE_PATH%" (
    CALL "%VENV_ACTIVATE_PATH%"
    IF ERRORLEVEL 1 (
        ECHO WARNING: Failed to activate venv, proceeding.
    )
)
ECHO.

IF NOT EXIST "%CHECKPOINT_OUTPUT_DIR%" MKDIR "%CHECKPOINT_OUTPUT_DIR%"
IF NOT EXIST "%DATA_DIR_BASE%" MKDIR "%DATA_DIR_BASE%"
IF NOT EXIST "%VIDEO_DATA_PATH%" MKDIR "%VIDEO_DATA_PATH%"
ECHO.

ECHO Starting training script: %SCRIPT_NAME%
ECHO.

IF %NPROC_PER_NODE% EQU 1 (
    "%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
) ELSE (
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
IF DEFINED PROMPT_AFTER_RUN ( PAUSE ) ELSE ( TIMEOUT /T 25 /NOBREAK >nul )
ENDLOCAL