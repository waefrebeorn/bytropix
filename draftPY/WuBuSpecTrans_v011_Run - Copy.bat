@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM --- Core Paths (Modify if your project structure is different) ---
SET "PROJECT_ROOT=%~dp0.."
SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT_NAME=WuBuSpecTrans_v0.1.1.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"
SET "DATA_DIR_BASE=%PROJECT_ROOT%\data"
SET "CHECKPOINT_OUTPUT_DIR_BASE=%PROJECT_ROOT%\checkpoints"
SET "RUN_NAME_SUFFIX=Run_Latest"
SET "CHECKPOINT_OUTPUT_DIR=%CHECKPOINT_OUTPUT_DIR_BASE%\WuBuSpecTrans_v011_%RUN_NAME_SUFFIX%"

REM --- Find Python Executable ---
IF EXIST "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
) ELSE IF EXIST "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) ELSE (
    SET "PYTHON_EXE=python"
)

REM --- DDP Configuration ---
SET "NPROC_PER_NODE=1"
SET "MASTER_ADDR=localhost"
SET "MASTER_PORT=29515"

REM --- Automatically Find Checkpoint to Load (for resuming) ---
SET "LOAD_CHECKPOINT="
SET "BEST_CKPT_NAME=wubuspectrans_ckpt_v011_best.pt"
SET "LATEST_EPOCH_CKPT_NAME="

IF EXIST "%CHECKPOINT_OUTPUT_DIR%\%BEST_CKPT_NAME%" (
    SET "LOAD_CHECKPOINT=%CHECKPOINT_OUTPUT_DIR%\%BEST_CKPT_NAME%"
    ECHO Found best checkpoint to potentially resume from: !LOAD_CHECKPOINT!
) ELSE (
    ECHO Best checkpoint not found. Searching for latest epoch/step checkpoint...
    FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "%CHECKPOINT_OUTPUT_DIR%\wubuspectrans_ckpt_v011_ep*_step*.pt"') DO (
        SET "LATEST_EPOCH_CKPT_NAME=%%F"
        GOTO FoundLatestEpochCkpt
    )
    :FoundLatestEpochCkpt
    IF DEFINED LATEST_EPOCH_CKPT_NAME (
        SET "LOAD_CHECKPOINT=%CHECKPOINT_OUTPUT_DIR%\!LATEST_EPOCH_CKPT_NAME!"
        ECHO Found latest epoch checkpoint to potentially resume from: !LOAD_CHECKPOINT!
    ) ELSE (
        ECHO No suitable checkpoint found in %CHECKPOINT_OUTPUT_DIR% to resume from. Will start a new run.
    )
)

REM --- Training Configuration ---
SET "AUDIO_DATA_PATH=%DATA_DIR_BASE%\demo_audio_data_dir"
SET "VALIDATION_AUDIO_PATH="

SET "LATENT_DIM=512"
SET "ENCODER_INITIAL_TANGENT_DIM=256"

SET "GAAD_NUM_REGIONS=64"
SET "GAAD_DECOMP_TYPE=hybrid"
SET "GAAD_MIN_SIZE_PX=4"

SET "REGION_PROC_SIZE_T=32"
SET "REGION_PROC_SIZE_F=32"
SET "DCT_NORM_TYPE=global_scale"
SET "DCT_NORM_GLOBAL_SCALE=100.0"
SET "DCT_NORM_TANH_SCALE=50.0"

REM ## Set the primary discriminator type for a new run, or if not switching ##
SET "DISCRIMINATOR_INPUT_TYPE=dct" 
REM SET "DISCRIMINATOR_INPUT_TYPE=mel" 

SET "DISC_APPLY_SPECTRAL_NORM=true"
SET "DISC_BASE_DISC_CHANNELS=64"
SET "DISC_MAX_DISC_CHANNELS=512"

SET "WUBU_DROPOUT=0.1"

SET "WUBU_S_NUM_LEVELS=3"
SET "WUBU_S_HYPERBOLIC_DIMS=128 64 32"
SET "WUBU_S_INITIAL_CURVATURES=1.0 0.8 0.6"
SET "WUBU_S_USE_ROTATION=false"
SET "WUBU_S_PHI_CURVATURE=true"
SET "WUBU_S_PHI_ROT_INIT=false"
SET "WUBU_S_OUTPUT_DIM_ENCODER=256"

SET "WUBU_G_NUM_LEVELS=3"
SET "WUBU_G_HYPERBOLIC_DIMS=64 128 256"
SET "WUBU_G_INITIAL_CURVATURES=0.6 0.8 1.0"
SET "WUBU_G_USE_ROTATION=false"
SET "WUBU_G_PHI_CURVATURE=true"
SET "WUBU_G_PHI_ROT_INIT=false"

REM ## These WuBu-D params configure the *primary* DCT-based D if DISCRIMINATOR_INPUT_TYPE=dct ##
REM ## If DISCRIMINATOR_INPUT_TYPE=mel, these are still passed but might not be used by the primary D ##
SET "WUBU_D_NUM_LEVELS=2"
SET "WUBU_D_HYPERBOLIC_DIMS=128 64"
SET "WUBU_D_INITIAL_CURVATURES=0.9 0.7"
SET "WUBU_D_USE_ROTATION=false"
SET "WUBU_D_PHI_CURVATURE=true"
SET "WUBU_D_PHI_ROT_INIT=false"
SET "WUBU_D_OUTPUT_DIM=64"

SET "EPOCHS=1500"
SET "GLOBAL_BATCH_SIZE=64"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
SET "LEARNING_RATE_GEN=1e-4"
SET "LEARNING_RATE_DISC=1e-4"
SET "RISGD_MAX_GRAD_NORM=1.0"
SET "GLOBAL_MAX_GRAD_NORM=5.0"
SET "TRAIN_LOG_INTERVAL=1"
SET "TRAIN_SAVE_INTERVAL=2000" 
SET "TRAIN_SEED=42"
SET "TRAIN_NUM_WORKERS=2"
SET "USE_AMP=true"
SET "DETECT_ANOMALY=false"
SET "LOAD_STRICT=true"

SET "LAMBDA_RECON=10.0"
SET "LAMBDA_KL=0.01"
SET "LAMBDA_KL_UPDATE_INTERVAL=20"
SET "MIN_LAMBDA_KL_Q_CONTROL=1e-3"
SET "MAX_LAMBDA_KL_Q_CONTROL=2.0"
SET "LAMBDA_GAN=1.0"

SET "SAMPLE_RATE=44100"
SET "N_FFT=1024"
SET "HOP_LENGTH=256"
SET "N_MELS=128"
SET "FMIN=0.0"
SET "FMAX="
SET "SEGMENT_DURATION_SEC=1.0"
SET "SEGMENT_OVERLAP_SEC=0.05"
SET "DB_NORM_MIN=-80.0"
SET "DB_NORM_MAX=0.0"
SET "PRELOAD_AUDIO_DATASET_TO_RAM=true"

SET "USE_LPIPS_FOR_MEL_VERIFICATION=true"
SET "VALIDATION_SPLIT_FRACTION=0.1"
SET "VAL_PRIMARY_METRIC=avg_val_recon_dct_mse"
SET "NUM_VAL_SAMPLES_TO_LOG=2"
SET "DEMO_NUM_SAMPLES=4"

SET "Q_CONTROLLER_ENABLED=true"
SET "WANDB_ENABLED=true"
SET "WANDB_PROJECT=WuBuSpecTransV011"
SET "WANDB_RUN_NAME="
SET "WANDB_LOG_TRAIN_RECON_INTERVAL=50"
SET "WANDB_LOG_FIXED_NOISE_INTERVAL=100"

SET "RESET_Q_CONTROLLERS_ON_LOAD=false"
REM SET "RESET_Q_CONTROLLERS_ON_LOAD=true" 

SET "ENABLE_HEURISTIC_DISC_SWITCHING=true"
SET "INITIAL_DISC_TYPE=%DISCRIMINATOR_INPUT_TYPE%" 
SET "DISC_SWITCH_CHECK_INTERVAL=40"
SET "DISC_SWITCH_MIN_STEPS_BETWEEN=200"
SET "DISC_SWITCH_PROBLEM_STATE_COUNT_THRESH=3"


SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --audio_dir_path "%AUDIO_DATA_PATH%""
IF DEFINED VALIDATION_AUDIO_PATH (
    IF NOT "%VALIDATION_AUDIO_PATH%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_audio_dir_path "%VALIDATION_AUDIO_PATH%""
)
IF DEFINED LOAD_CHECKPOINT (
    IF NOT "%LOAD_CHECKPOINT%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --load_checkpoint "%LOAD_CHECKPOINT%""
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR%""

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --sample_rate %SAMPLE_RATE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --n_fft %N_FFT%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --hop_length %HOP_LENGTH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --n_mels %N_MELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --fmin %FMIN%"
IF DEFINED FMAX ( IF NOT "%FMAX%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --fmax %FMAX%" )
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --segment_duration_sec %SEGMENT_DURATION_SEC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --segment_overlap_sec %SEGMENT_OVERLAP_SEC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --db_norm_min %DB_NORM_MIN%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --db_norm_max %DB_NORM_MAX%"
IF /I "%PRELOAD_AUDIO_DATASET_TO_RAM%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --preload_audio_dataset_to_ram"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --latent_dim %LATENT_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --encoder_initial_tangent_dim %ENCODER_INITIAL_TANGENT_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_num_regions %GAAD_NUM_REGIONS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_decomposition_type %GAAD_DECOMP_TYPE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_min_size_px %GAAD_MIN_SIZE_PX%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --region_proc_size_t %REGION_PROC_SIZE_T%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --region_proc_size_f %REGION_PROC_SIZE_F%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --dct_norm_type %DCT_NORM_TYPE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --dct_norm_global_scale %DCT_NORM_GLOBAL_SCALE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --dct_norm_tanh_scale %DCT_NORM_TANH_SCALE%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_input_type %DISCRIMINATOR_INPUT_TYPE%"
IF /I "%DISC_APPLY_SPECTRAL_NORM%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_apply_spectral_norm"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_base_disc_channels %DISC_BASE_DISC_CHANNELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_max_disc_channels %DISC_MAX_DISC_CHANNELS%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_dropout %WUBU_DROPOUT%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_num_levels %WUBU_S_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_hyperbolic_dims %WUBU_S_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_initial_curvatures %WUBU_S_INITIAL_CURVATURES%"
IF /I "%WUBU_S_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_use_rotation"
IF /I "%WUBU_S_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_curvature"
IF /I "%WUBU_S_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_rotation_init"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_output_dim_encoder %WUBU_S_OUTPUT_DIM_ENCODER%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_g_num_levels %WUBU_G_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_g_hyperbolic_dims %WUBU_G_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_g_initial_curvatures %WUBU_G_INITIAL_CURVATURES%"
IF /I "%WUBU_G_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_use_rotation"
IF /I "%WUBU_G_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_phi_influence_curvature"
IF /I "%WUBU_G_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_phi_influence_rotation_init"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_num_levels %WUBU_D_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_hyperbolic_dims %WUBU_D_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_initial_curvatures %WUBU_D_INITIAL_CURVATURES%"
IF /I "%WUBU_D_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_use_rotation"
IF /I "%WUBU_D_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_phi_influence_curvature"
IF /I "%WUBU_D_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_phi_influence_rotation_init"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_output_dim %WUBU_D_OUTPUT_DIM%"

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
IF /I "%LOAD_STRICT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_strict"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_recon %LAMBDA_RECON%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_kl %LAMBDA_KL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_kl_update_interval %LAMBDA_KL_UPDATE_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --min_lambda_kl_q_control %MIN_LAMBDA_KL_Q_CONTROL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --max_lambda_kl_q_control %MAX_LAMBDA_KL_Q_CONTROL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_gan %LAMBDA_GAN%"

IF /I "%USE_LPIPS_FOR_MEL_VERIFICATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_lpips_for_mel_verification"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_split_fraction %VALIDATION_SPLIT_FRACTION%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_primary_metric %VAL_PRIMARY_METRIC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_val_samples_to_log %NUM_VAL_SAMPLES_TO_LOG%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --demo_num_samples %DEMO_NUM_SAMPLES%"

IF /I "%Q_CONTROLLER_ENABLED%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled"
IF /I "%RESET_Q_CONTROLLERS_ON_LOAD%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --reset_q_controllers_on_load"

IF /I "%ENABLE_HEURISTIC_DISC_SWITCHING%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --enable_heuristic_disc_switching"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --initial_disc_type %INITIAL_DISC_TYPE%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_switch_check_interval %DISC_SWITCH_CHECK_INTERVAL%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_switch_min_steps_between %DISC_SWITCH_MIN_STEPS_BETWEEN%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_switch_problem_state_count_thresh %DISC_SWITCH_PROBLEM_STATE_COUNT_THRESH%"
)

IF /I "%WANDB_ENABLED%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb --wandb_project %WANDB_PROJECT%"
    IF DEFINED WANDB_RUN_NAME (
        IF NOT "%WANDB_RUN_NAME%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_run_name "%WANDB_RUN_NAME%""
    )
)
IF %WANDB_LOG_TRAIN_RECON_INTERVAL% GTR 0 SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_log_train_recon_interval %WANDB_LOG_TRAIN_RECON_INTERVAL%"
IF %WANDB_LOG_FIXED_NOISE_INTERVAL% GTR 0 SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_log_fixed_noise_samples_interval %WANDB_LOG_FIXED_NOISE_INTERVAL%"

ECHO ======================================================
ECHO WuBuSpecTrans VAE-GAN - Audio Run
ECHO Python: %PYTHON_EXE%
ECHO Script Name: %SCRIPT_NAME%
ECHO Checkpoints: %CHECKPOINT_OUTPUT_DIR%
IF DEFINED LOAD_CHECKPOINT (
    IF NOT "%LOAD_CHECKPOINT%"=="" ECHO Loading Checkpoint: %LOAD_CHECKPOINT%
)
ECHO Audio Data: %AUDIO_DATA_PATH%
ECHO Primary Discriminator Config Type: %DISCRIMINATOR_INPUT_TYPE%
ECHO Heuristic D Switching Enabled: %ENABLE_HEURISTIC_DISC_SWITCHING%
IF /I "%ENABLE_HEURISTIC_DISC_SWITCHING%"=="true" ECHO Initial Active D Type for Switching: %INITIAL_DISC_TYPE%
ECHO AMP: %USE_AMP%
ECHO Q-Controller: %Q_CONTROLLER_ENABLED%
ECHO Reset Q-Controllers on Load: %RESET_Q_CONTROLLERS_ON_LOAD%
ECHO Learning Rate (Gen/Disc): %LEARNING_RATE_GEN% / %LEARNING_RATE_DISC%
ECHO Batch Size (Global/PerGPU): %GLOBAL_BATCH_SIZE% / %BATCH_SIZE_PER_GPU%
ECHO Grad Accum Steps: %GRAD_ACCUM_STEPS%
ECHO ======================================================
ECHO.

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
IF NOT EXIST "%AUDIO_DATA_PATH%" MKDIR "%AUDIO_DATA_PATH%"
ECHO.

ECHO Starting training script: %SCRIPT_NAME%
ECHO.
ECHO Full command:
ECHO "%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
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