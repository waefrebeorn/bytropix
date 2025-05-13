@echo OFF
REM ==================================================================================
REM Batch file for WuBuNestDiffusion_v0.10.1_OpticalFlow - Run 1
REM Goal: Enable all features, including Optical Flow motion branch.
REM Uses DELAYED EXPANSION for SCRIPT_ARGS within conditional blocks.
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
SET "SCRIPT_NAME=WuBuNestDiffusion_v0.10.1_OpticalFlow.py" 
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

REM --- DDP Settings ---
SET "NPROC_PER_NODE=1"
SET "MASTER_ADDR=localhost"
SET "MASTER_PORT=29511"  REM <-- Incremented port just in case

REM --- Data and Output Directories ---
SET "DATA_DIR_BASE=%PROJECT_ROOT%\data"
SET "CHECKPOINT_OUTPUT_DIR=%PROJECT_ROOT%\checkpoints\WuBuRegional_v0101_Flow_Run1_AllFeatures_Anomaly" 
SET "VIDEO_DATA_PATH=%DATA_DIR_BASE%\demo_video_data_dir"
SET "VALIDATION_VIDEO_PATH="

REM --- Model & Training Hyperparameters ---
SET "IMAGE_H=176"
SET "IMAGE_W=320"
SET "NUM_CHANNELS=3"
SET "NUM_INPUT_FRAMES=10"
SET "NUM_PREDICT_FRAMES=2"
SET "FRAME_SKIP=1"

REM --- GAAD Parameters (Encoder/Decoder Definition) ---
SET "GAAD_NUM_REGIONS=24"
SET "GAAD_DECOMP_TYPE=hybrid"
SET "GAAD_MIN_SIZE_PX=5"

REM --- Encoder Parameters ---
SET "ENCODER_USE_ROI_ALIGN=true"
SET "ENCODER_PIXEL_PATCH_SIZE=16"
SET "ENCODER_SHALLOW_CNN_CHANNELS=32"
SET "ENCODER_ROI_ALIGN_OUTPUT_H=4"
SET "ENCODER_ROI_ALIGN_OUTPUT_W=4"
SET "ENCODER_INITIAL_TANGENT_DIM=192"

REM --- Decoder Parameters ---
SET "DECODER_TYPE=patch_gen"
SET "DECODER_PATCH_GEN_SIZE=16"
SET "DECODER_PATCH_RESIZE_MODE=bilinear"
SET "DECODER_USE_TARGET_GT_BBOXES_FOR_SAMPLING=false"

REM --- Diffusion Parameters ---
SET "TIMESTEPS=150"
SET "DIFFUSION_TIME_EMBEDDING_DIM=192"
SET "BETA_SCHEDULE=cosine"
SET "COSINE_S=0.008"
SET "PHI_TIME_DIFFUSION_SCALE=1.0"
SET "PHI_TIME_BASE_FREQ=10000.0"
SET "USE_PHI_FREQUENCY_SCALING_FOR_TIME_EMB=true"
SET "LOSS_TYPE=pixel_reconstruction"

REM --- WuBu Parameters ---
SET "WUBU_S_NUM_LEVELS=3"
SET "WUBU_S_HYPERBOLIC_DIMS=96 64 48"
SET "WUBU_S_INITIAL_CURVATURES=1.0 0.7 0.5"
SET "WUBU_S_USE_ROTATION=false"
SET "WUBU_S_PHI_CURVATURE=true"
SET "WUBU_S_PHI_ROT_INIT=false"

SET "WUBU_T_NUM_LEVELS=3"
SET "WUBU_T_HYPERBOLIC_DIMS=192 128 96"
SET "WUBU_T_INITIAL_CURVATURES=1.0 0.7 0.5"
SET "WUBU_T_USE_ROTATION=false"
SET "WUBU_T_PHI_CURVATURE=true"
SET "WUBU_T_PHI_ROT_INIT=false"

SET "WUBU_DROPOUT=0.1"

REM --- Motion Branch Parameters (Enabled, Now uses Optical Flow) ---
SET "USE_WUBU_MOTION_BRANCH=true"
SET "GAAD_MOTION_NUM_REGIONS=16"
SET "GAAD_MOTION_DECOMP_TYPE=hybrid"
SET "WUBU_M_NUM_LEVELS=3"
SET "WUBU_M_HYPERBOLIC_DIMS=128 96 64"
SET "WUBU_M_INITIAL_CURVATURES=1.0 0.8 0.6"
SET "WUBU_M_USE_ROTATION=false"
SET "WUBU_M_PHI_CURVATURE=true"
SET "WUBU_M_PHI_ROT_INIT=false"

REM --- Motion Encoder Optical Flow Parameters (NEW) ---
SET "OPTICAL_FLOW_NET_TYPE=raft_small" 
REM Choices: raft_small, raft_large (if available)
SET "FREEZE_FLOW_NET=true"            
REM Freeze pre-trained weights? true/false
SET "FLOW_STATS_COMPONENTS=mag_mean angle_mean" 
REM Stats to embed. E.g., "mag_mean angle_mean mag_std angle_std"

REM --- Transformer Noise Predictor Parameters ---
SET "TNP_NUM_LAYERS=4"
SET "TNP_NUM_HEADS=8"
SET "TNP_D_MODEL=256"
SET "TNP_D_FF_RATIO=4.0"
SET "TNP_DROPOUT=0.1"

REM --- Training Parameters ---
SET "EPOCHS=5"
SET "GLOBAL_BATCH_SIZE=12"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
SET "LEARNING_RATE=5e-5"
SET "RISGD_MAX_GRAD_NORM=5.0"
SET "GLOBAL_MAX_GRAD_NORM=6.0"
SET "TRAIN_LOG_INTERVAL=2"
SET "TRAIN_SAVE_INTERVAL=50000000"
SET "TRAIN_SEED=42"
SET "TRAIN_NUM_WORKERS=4"
REM SET "USE_AMP=true"              
REM SET "DETECT_ANOMALY=true"       

REM --- CFG, Sampler, Validation ---
SET "CFG_UNCONDITIONAL_DROPOUT_PROB=0.1"

SET "VAL_CFG_SCALE=1.5"
SET "VAL_SAMPLER_TYPE=ddim"
SET "VAL_SAMPLING_STEPS=20"
SET "DDIM_X0_CLIP_VAL=1.0"
SET "USE_LPIPS_FOR_VERIFICATION=false"
SET "VALIDATION_SPLIT_FRACTION=0.1"
SET "VAL_PRIMARY_METRIC=avg_val_pixel_mse"
SET "NUM_VAL_SAMPLES_TO_LOG=2"
SET "DEMO_SAMPLER_TYPE=ddim"
SET "DEMO_DDIM_ETA=0.0"
SET "DEMO_CFG_SCALE=3.0"
SET "DEMO_SAMPLING_STEPS=50"

REM --- Script Execution Arguments Construction ---
SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --video_data_path "%VIDEO_DATA_PATH%""
IF DEFINED VALIDATION_VIDEO_PATH (
    IF NOT "%VALIDATION_VIDEO_PATH%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_video_path "%VALIDATION_VIDEO_PATH%""
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR%""
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --image_h %IMAGE_H%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --image_w %IMAGE_W%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_channels %NUM_CHANNELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_input_frames %NUM_INPUT_FRAMES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_predict_frames %NUM_PREDICT_FRAMES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --frame_skip %FRAME_SKIP%"

REM GAAD Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_num_regions %GAAD_NUM_REGIONS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_decomposition_type %GAAD_DECOMP_TYPE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_min_size_px %GAAD_MIN_SIZE_PX%"

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

REM Decoder Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --decoder_type %DECODER_TYPE%"
IF /I "%DECODER_TYPE%"=="patch_gen" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --decoder_patch_gen_size %DECODER_PATCH_GEN_SIZE%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --decoder_patch_resize_mode %DECODER_PATCH_RESIZE_MODE%"
)
IF /I "%DECODER_USE_TARGET_GT_BBOXES_FOR_SAMPLING%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --decoder_use_target_gt_bboxes_for_sampling"


REM Diffusion Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --timesteps %TIMESTEPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --diffusion_time_embedding_dim %DIFFUSION_TIME_EMBEDDING_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --beta_schedule %BETA_SCHEDULE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --cosine_s %COSINE_S%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --phi_time_diffusion_scale %PHI_TIME_DIFFUSION_SCALE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --phi_time_base_freq %PHI_TIME_BASE_FREQ%"
IF /I "%USE_PHI_FREQUENCY_SCALING_FOR_TIME_EMB%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --use_phi_frequency_scaling_for_time_emb"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --loss_type %LOSS_TYPE%"

REM WuBu-S Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_num_levels %WUBU_S_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_hyperbolic_dims %WUBU_S_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_initial_curvatures %WUBU_S_INITIAL_CURVATURES%"
IF /I "%WUBU_S_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_use_rotation"
IF /I "%WUBU_S_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_phi_influence_curvature"
IF /I "%WUBU_S_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_phi_influence_rotation_init"

REM WuBu-T Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_num_levels %WUBU_T_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_hyperbolic_dims %WUBU_T_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_initial_curvatures %WUBU_T_INITIAL_CURVATURES%"
IF /I "%WUBU_T_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_use_rotation"
IF /I "%WUBU_T_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_phi_influence_curvature"
IF /I "%WUBU_T_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_phi_influence_rotation_init"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_dropout %WUBU_DROPOUT%"

REM Motion Branch Arguments (Conditional)
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

    REM --> ADD NEW OPTICAL FLOW ARGS HERE <--
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optical_flow_net_type %OPTICAL_FLOW_NET_TYPE%"
    IF /I "%FREEZE_FLOW_NET%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --freeze_flow_net"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --flow_stats_components %FLOW_STATS_COMPONENTS%"

) ELSE (
    ECHO Motion Branch IS DISABLED - Skipping motion specific arguments
)

REM Transformer Noise Predictor Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --tnp_num_layers %TNP_NUM_LAYERS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --tnp_num_heads %TNP_NUM_HEADS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --tnp_d_model %TNP_D_MODEL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --tnp_d_ff_ratio %TNP_D_FF_RATIO%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --tnp_dropout %TNP_DROPOUT%"

REM Training Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --epochs %EPOCHS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --batch_size %BATCH_SIZE_PER_GPU%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --grad_accum_steps %GRAD_ACCUM_STEPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --learning_rate %LEARNING_RATE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --risgd_max_grad_norm %RISGD_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --global_max_grad_norm %GLOBAL_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --log_interval %TRAIN_LOG_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --save_interval %TRAIN_SAVE_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --seed %TRAIN_SEED%"
IF /I "%USE_AMP%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --use_amp"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_workers %TRAIN_NUM_WORKERS%"
IF /I "%DETECT_ANOMALY%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --detect_anomaly"

REM CFG, Sampler, Validation Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --cfg_unconditional_dropout_prob %CFG_UNCONDITIONAL_DROPOUT_PROB%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_cfg_scale %VAL_CFG_SCALE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_sampler_type %VAL_SAMPLER_TYPE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_sampling_steps %VAL_SAMPLING_STEPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --ddim_x0_clip_val %DDIM_X0_CLIP_VAL%"
IF /I "%USE_LPIPS_FOR_VERIFICATION%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --use_lpips_for_verification"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_split_fraction %VALIDATION_SPLIT_FRACTION%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_primary_metric %VAL_PRIMARY_METRIC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_val_samples_to_log %NUM_VAL_SAMPLES_TO_LOG%"

REM Demo Sampling Args
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --demo_sampler_type %DEMO_SAMPLER_TYPE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --demo_ddim_eta %DEMO_DDIM_ETA%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --demo_cfg_scale %DEMO_CFG_SCALE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --demo_sampling_steps %DEMO_SAMPLING_STEPS%"

REM Other Flags
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_controller_enabled"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wandb --wandb_project WuBuRegionalDiff_v0101_Flow_AllFeat_Anomaly" 


REM --- Environment Activation & Execution ---
ECHO ======================================================
ECHO WuBu GAAD Regional Latent Diffusion - V0.10.1 OpticalFlow RUN 1
ECHO Python: %PYTHON_EXE%
ECHO Script Name: %SCRIPT_NAME%
ECHO Checkpoints: %CHECKPOINT_OUTPUT_DIR%
ECHO Motion Branch Enabled: %USE_WUBU_MOTION_BRANCH% (using Optical Flow: %OPTICAL_FLOW_NET_TYPE%)
ECHO Anomaly Detection: %DETECT_ANOMALY%
ECHO AMP Enabled: %USE_AMP%
ECHO Encoder Type: %ENCODER_USE_ROI_ALIGN% (True=RoIAlign, False=PixelPatch)
ECHO Decoder Type: %DECODER_TYPE%
ECHO Learning Rate: %LEARNING_RATE%
ECHO Grad Clip (RISGD / Global): %RISGD_MAX_GRAD_NORM% / %GLOBAL_MAX_GRAD_NORM%
ECHO WuBu-S/T/M Levels: %WUBU_S_NUM_LEVELS% / %WUBU_T_NUM_LEVELS% / %WUBU_M_NUM_LEVELS%
ECHO Loss Type: %LOSS_TYPE%
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
ECHO.

ECHO Starting training script: %SCRIPT_NAME%
ECHO Script path: %FULL_SCRIPT_PATH%
ECHO.

ECHO == DEBUGGING BATCH VARS (Pre-execution check) ==
ECHO USE_WUBU_MOTION_BRANCH = %USE_WUBU_MOTION_BRANCH%
ECHO OPTICAL_FLOW_NET_TYPE = %OPTICAL_FLOW_NET_TYPE%
ECHO FREEZE_FLOW_NET = %FREEZE_FLOW_NET%
ECHO FLOW_STATS_COMPONENTS = %FLOW_STATS_COMPONENTS%
ECHO LOSS_TYPE = %LOSS_TYPE%
ECHO DECODER_TYPE = %DECODER_TYPE%
ECHO ENCODER_USE_ROI_ALIGN = %ENCODER_USE_ROI_ALIGN%
ECHO USE_AMP = %USE_AMP%
ECHO DETECT_ANOMALY = %DETECT_ANOMALY%
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
IF DEFINED PROMPT_AFTER_RUN ( PAUSE ) ELSE ( TIMEOUT /T 5 /NOBREAK >nul )
ENDLOCAL