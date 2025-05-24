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
SET "SCRIPT_NAME=WuBuGAADHybridGen_v0.3.py"
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
SET "CHECKPOINT_OUTPUT_DIR=%PROJECT_ROOT%\checkpoints\WuBuGAADHybridGen_v03_Run_Latest"
SET "VIDEO_DATA_PATH=%DATA_DIR_BASE%\demo_video_data_dir_dft_dct"
SET "VALIDATION_VIDEO_PATH="
SET "LOAD_CHECKPOINT="
SET "BEST_CKPT_NAME=wubugaad_hybridgen_v03_dft_dct_best.pt"
SET "LATEST_EPOCH_CKPT_NAME="

IF EXIST "%CHECKPOINT_OUTPUT_DIR%\%BEST_CKPT_NAME%" (
    SET "LOAD_CHECKPOINT=%CHECKPOINT_OUTPUT_DIR%\%BEST_CKPT_NAME%"
) ELSE (
    FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "%CHECKPOINT_OUTPUT_DIR%\wubugaad_hybridgen_v03_dft_dct_ep*_step*.pt"') DO (
        SET "LATEST_EPOCH_CKPT_NAME=%%F"
        GOTO FoundLatestEpochCkpt_AllInDFTv2
    )
    :FoundLatestEpochCkpt_AllInDFTv2
    IF DEFINED LATEST_EPOCH_CKPT_NAME (
        SET "LOAD_CHECKPOINT=%CHECKPOINT_OUTPUT_DIR%\!LATEST_EPOCH_CKPT_NAME!"
    )
)
REM =====================================================================
REM Data and Model Core Configuration
REM =====================================================================
SET "IMAGE_H=256"
SET "IMAGE_W=256"
SET "NUM_CHANNELS=3"
SET "NUM_INPUT_FRAMES=15"
SET "NUM_PREDICT_FRAMES=5"
SET "FRAME_SKIP=1"
SET "LATENT_DIM=1024"

REM =====================================================================
REM GAAD Configuration
REM =====================================================================
SET "GAAD_NUM_REGIONS=32"
SET "GAAD_DECOMP_TYPE=hybrid"
SET "GAAD_MIN_SIZE_PX=16"

REM =====================================================================
REM Spectral Transforms Configuration (DFT + DCT for v0.3)
REM =====================================================================
SET "USE_DFT_FEATURES_APPEARANCE=true"
SET "USE_DCT_FEATURES_APPEARANCE=true"
SET "SPECTRAL_PATCH_SIZE_H=16"
SET "SPECTRAL_PATCH_SIZE_W=16"
SET "DFT_NORM_SCALE_VIDEO=25.0"
SET "DFT_FFT_NORM=ortho"
SET "DCT_NORM_TYPE=tanh"
SET "DCT_NORM_GLOBAL_SCALE=150.0"
SET "DCT_NORM_TANH_SCALE=40.0"

REM =====================================================================
REM Encoder Architecture Configuration
REM =====================================================================
SET "ENCODER_USE_ROI_ALIGN=true"
SET "ENCODER_SHALLOW_CNN_CHANNELS=64"
SET "ENCODER_INITIAL_TANGENT_DIM=256"

REM =====================================================================
REM Generator Architecture Configuration
REM =====================================================================
SET "GEN_TEMPORAL_KERNEL_SIZE=5"
SET "GEN_FINAL_CONV_KERNEL_SPATIAL=5"
SET "GEN_USE_GAAD_FILM_CONDITION=true"

REM =====================================================================
REM General Discriminator Configuration (Defaults for variants)
REM =====================================================================
SET "DISC_APPLY_SPECTRAL_NORM=true"
SET "DISC_BASE_DISC_CHANNELS=64"
SET "DISC_MAX_DISC_CHANNELS=1024"
SET "DISC_TEMPORAL_KERNEL_SIZE=5"
SET "DISC_TARGET_FINAL_FEATURE_DIM=4 4"
SET "MAX_VIDEO_DISC_DOWNSAMPLE_LAYERS=5"
SET "DISC_USE_GAAD_FILM_CONDITION=true"
SET "DISC_GAAD_CONDITION_DIM_DISC=128"
SET "DISC_MIN_HIDDEN_FC_DIM=256"
SET "DISC_MAX_HIDDEN_FC_DIM=1024"

REM =====================================================================
REM Discriminator Architecture Variants & Switching Configuration
REM =====================================================================
SET "PRIMARY_DISC_ARCHITECTURE_VARIANT=global_wubu_video_feature"
SET "ALT_DISC_ARCHITECTURE_VARIANT=default_pixel_cnn"
REM global_wubu_video_feature or default_pixel_cnn
SET "ENABLE_HEURISTIC_DISC_SWITCHING=true"
SET "INITIAL_DISC_TYPE=feature"
REM pixel or feature
SET "DISC_SWITCH_CHECK_INTERVAL=25"
SET "DISC_SWITCH_MIN_STEPS_BETWEEN=50"
SET "DISC_SWITCH_PROBLEM_STATE_COUNT_THRESH=2"

REM =====================================================================
REM GlobalWuBuVideoFeatureDiscriminator Specifics
REM =====================================================================
SET "VIDEO_GLOBAL_WUBU_D_INPUT_TANGENT_DIM=512"
SET "VIDEO_GLOBAL_WUBU_D_OUTPUT_FEATURE_DIM=256"
SET "DISC_USE_GLOBAL_STATS_AUX_VIDEO_GLOBAL_WUBU=true"
SET "DISC_GLOBAL_STATS_MLP_HIDDEN_DIM_VIDEO_GLOBAL_WUBU=64"

REM =====================================================================
REM WuBu Configuration (Common)
REM =====================================================================
SET "WUBU_DROPOUT=0.05"

REM =====================================================================
REM WuBu-S (Appearance) Configuration
REM =====================================================================
SET "WUBU_S_NUM_LEVELS=3"
SET "WUBU_S_HYPERBOLIC_DIMS=512 256 128"
SET "WUBU_S_INITIAL_CURVATURES=1.0 0.8 0.6"
SET "WUBU_S_INITIAL_SCALES=1.0"
SET "WUBU_S_INITIAL_SPREAD_VALUES=0.1"
SET "WUBU_S_BOUNDARY_POINTS_PER_LEVEL=4"
SET "WUBU_S_USE_ROTATION=true"
SET "WUBU_S_PHI_INFLUENCE_CURVATURE=true"
SET "WUBU_S_PHI_INFLUENCE_ROTATION_INIT=true"

REM =====================================================================
REM WuBu-M (Motion) Configuration
REM =====================================================================
SET "USE_WUBU_MOTION_BRANCH=true"
SET "GAAD_MOTION_NUM_REGIONS=24"
SET "GAAD_MOTION_DECOMP_TYPE=hybrid"
SET "WUBU_M_NUM_LEVELS=3"
SET "WUBU_M_HYPERBOLIC_DIMS=256 128 64"
SET "WUBU_M_INITIAL_CURVATURES=1.0 0.8 0.7"
SET "WUBU_M_INITIAL_SCALES=1.0"
SET "WUBU_M_INITIAL_SPREAD_VALUES=0.1"
SET "WUBU_M_BOUNDARY_POINTS_PER_LEVEL=0"
SET "WUBU_M_USE_ROTATION=true"
SET "WUBU_M_PHI_INFLUENCE_CURVATURE=true"
SET "WUBU_M_PHI_INFLUENCE_ROTATION_INIT=true"
SET "OPTICAL_FLOW_NET_TYPE=raft_large"
SET "FREEZE_FLOW_NET=true"
SET "FLOW_STATS_COMPONENTS=mag_mean angle_mean mag_std angle_std"

REM =====================================================================
REM WuBu-T (Temporal Aggregation) Configuration
REM =====================================================================
SET "WUBU_T_NUM_LEVELS=3"
SET "WUBU_T_HYPERBOLIC_DIMS=512 256 128"
SET "WUBU_T_INITIAL_CURVATURES=1.0 0.8 0.6"
SET "WUBU_T_INITIAL_SCALES=1.0"
SET "WUBU_T_INITIAL_SPREAD_VALUES=0.1"
SET "WUBU_T_BOUNDARY_POINTS_PER_LEVEL=0"
SET "WUBU_T_USE_ROTATION=true"
SET "WUBU_T_PHI_INFLUENCE_CURVATURE=true"
SET "WUBU_T_PHI_INFLUENCE_ROTATION_INIT=true"

REM =====================================================================
REM WuBu-D-Global-Video (Feature Discriminator) Configuration
REM =====================================================================
SET "WUBU_D_GLOBAL_VIDEO_NUM_LEVELS=3"
SET "WUBU_D_GLOBAL_VIDEO_HYPERBOLIC_DIMS=256 128 64"
SET "WUBU_D_GLOBAL_VIDEO_INITIAL_CURVATURES=0.8 0.6 0.4"
SET "WUBU_D_GLOBAL_VIDEO_INITIAL_SCALES=1.0"
SET "WUBU_D_GLOBAL_VIDEO_INITIAL_SPREAD_VALUES=0.1"
SET "WUBU_D_GLOBAL_VIDEO_BOUNDARY_POINTS_PER_LEVEL=0"
SET "WUBU_D_GLOBAL_VIDEO_USE_ROTATION=true"
SET "WUBU_D_GLOBAL_VIDEO_PHI_INFLUENCE_CURVATURE=true"
SET "WUBU_D_GLOBAL_VIDEO_PHI_INFLUENCE_ROTATION_INIT=true"

REM =====================================================================
REM Training Hyperparameters
REM =====================================================================
SET "EPOCHS=5000"
SET "GLOBAL_BATCH_SIZE=1"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
SET "LEARNING_RATE_GEN=1e-4"
SET "LEARNING_RATE_DISC=3e-5"
SET "LEARNING_RATE_DISC_ALT=3e-5"
SET "RISGD_MAX_GRAD_NORM=2.0"
SET "GLOBAL_MAX_GRAD_NORM=2.0"
SET "TRAIN_SEED=1337"
SET "TRAIN_NUM_WORKERS=4"
SET "USE_AMP=true"
SET "DETECT_ANOMALY=false"
SET "LOAD_STRICT=true"

REM =====================================================================
REM Loss Weights
REM =====================================================================
SET "LAMBDA_RECON=10.0"
SET "LAMBDA_RECON_DFT=7.0"
SET "LAMBDA_RECON_DCT=7.0"
SET "LAMBDA_KL=0.01"
SET "LAMBDA_GAN=1.0"

REM =====================================================================
REM Q-Controller for Lambda_KL (Scheduled) & Heuristics
REM =====================================================================
SET "Q_CONTROLLER_ENABLED=true"
SET "RESET_Q_CONTROLLERS_ON_LOAD=true"
SET "RESET_LKL_Q_CONTROLLER_ON_LOAD=true"
SET "LAMBDA_KL_UPDATE_INTERVAL=25"
SET "MIN_LAMBDA_KL_Q_CONTROL=1e-7"
SET "MAX_LAMBDA_KL_Q_CONTROL=0.1"
SET "Q_LKL_SCALE_OPTIONS=0.75 0.9 1.0 1.1 1.25"
SET "Q_LKL_LR_MOM_PROBATION_STEPS=15"
SET "Q_LKL_ACTION_PROBATION_STEPS=15"

SET "ENABLE_HEURISTIC_INTERVENTIONS=true"
SET "HEURISTIC_CHECK_INTERVAL=10"
SET "HEURISTIC_SHORT_TERM_HISTORY_LEN=7"
SET "HEURISTIC_TRIGGER_COUNT_THRESH=2"
SET "HEURISTIC_D_STRONG_THRESH=0.20"
SET "HEURISTIC_D_WEAK_THRESH=1.2"
SET "HEURISTIC_D_VERY_WEAK_THRESH=2.0"
SET "HEURISTIC_G_STALLED_THRESH=1.8"
SET "HEURISTIC_G_WINNING_THRESH=0.15"
SET "HEURISTIC_G_VERY_MUCH_WINNING_THRESH=0.03"
SET "HEURISTIC_KL_HIGH_THRESH=15.0"
SET "HEURISTIC_RECON_STAGNATION_IMPROVEMENT_THRESH_REL=0.001"
SET "TARGET_GOOD_RECON_THRESH_HEURISTIC_VIDEO=0.015"
SET "HEURISTIC_Q_REWARD_STAGNATION_THRESH=-0.3"
SET "HEURISTIC_RECON_BOOST_FACTOR_VIDEO=1.5"
SET "LAMBDA_FEAT_MATCH_HEURISTIC_VIDEO=0.1"
SET "LAMBDA_G_EASY_WIN_PENALTY_HEURISTIC_VIDEO=1.0"
SET "G_EASY_WIN_PENALTY_EPS_DENOM=1e-5"
SET "MAX_G_EASY_WIN_PENALTY_ABS=10.0"
SET "HEURISTIC_ACTIVE_D_LR_BOOST_FACTOR=2.0"
SET "HEURISTIC_D_Q_EXPLORE_BOOST_EPSILON=0.75"
SET "HEURISTIC_D_Q_EXPLORE_DURATION=15"
SET "HEURISTIC_MIN_LAMBDA_GAN_FACTOR=0.5"
SET "HEURISTIC_MAX_LAMBDA_GAN_FACTOR=1.5"
SET "FORCE_START_EPOCH_ON_LOAD="
SET "FORCE_START_GSTEP_ON_LOAD="

REM =====================================================================
REM Validation and Logging
REM =====================================================================
SET "LOG_INTERVAL=1"
SET "SAVE_INTERVAL=1000000"
SET "SAVE_EPOCH_INTERVAL=1"
SET "VALIDATION_INTERVAL_EPOCHS=1"
SET "DISABLE_VAL_TQDM=false"
SET "USE_LPIPS_FOR_VERIFICATION=true"
SET "VALIDATION_SPLIT_FRACTION=0.1"
SET "VAL_BLOCK_SIZE=20"
SET "VAL_PRIMARY_METRIC=avg_val_psnr"
SET "NUM_VAL_SAMPLES_TO_LOG=2"
SET "DEMO_NUM_SAMPLES=4"

REM =====================================================================
REM Experiment Control (WandB)
REM =====================================================================
SET "WANDB_ENABLED=true"
SET "WANDB_PROJECT=WuBuGAADHybridGenV03"
SET "WANDB_RUN_NAME="
SET "WANDB_LOG_TRAIN_RECON_INTERVAL=25"
SET "WANDB_LOG_FIXED_NOISE_SAMPLES_INTERVAL=100"

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

IF /I "%USE_DFT_FEATURES_APPEARANCE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_dft_features_appearance"
IF /I "%USE_DCT_FEATURES_APPEARANCE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_dct_features_appearance"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --spectral_patch_size_h %SPECTRAL_PATCH_SIZE_H%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --spectral_patch_size_w %SPECTRAL_PATCH_SIZE_W%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dft_norm_scale_video %DFT_NORM_SCALE_VIDEO%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dft_fft_norm %DFT_FFT_NORM%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dct_norm_type %DCT_NORM_TYPE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dct_norm_global_scale %DCT_NORM_GLOBAL_SCALE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dct_norm_tanh_scale %DCT_NORM_TANH_SCALE%"

IF /I "%ENCODER_USE_ROI_ALIGN%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_use_roi_align"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_shallow_cnn_channels %ENCODER_SHALLOW_CNN_CHANNELS%"
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --encoder_initial_tangent_dim %ENCODER_INITIAL_TANGENT_DIM%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gen_temporal_kernel_size %GEN_TEMPORAL_KERNEL_SIZE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gen_final_conv_kernel_spatial %GEN_FINAL_CONV_KERNEL_SPATIAL%"
IF /I "%GEN_USE_GAAD_FILM_CONDITION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gen_use_gaad_film_condition"

IF /I "%DISC_APPLY_SPECTRAL_NORM%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_apply_spectral_norm"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_base_disc_channels %DISC_BASE_DISC_CHANNELS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_max_disc_channels %DISC_MAX_DISC_CHANNELS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_temporal_kernel_size %DISC_TEMPORAL_KERNEL_SIZE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_target_final_feature_dim %DISC_TARGET_FINAL_FEATURE_DIM%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --max_video_disc_downsample_layers %MAX_VIDEO_DISC_DOWNSAMPLE_LAYERS%"
IF /I "%DISC_USE_GAAD_FILM_CONDITION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_use_gaad_film_condition"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_gaad_condition_dim_disc %DISC_GAAD_CONDITION_DIM_DISC%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --primary_disc_architecture_variant %PRIMARY_DISC_ARCHITECTURE_VARIANT%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --alt_disc_architecture_variant %ALT_DISC_ARCHITECTURE_VARIANT%"
IF /I "%ENABLE_HEURISTIC_DISC_SWITCHING%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --enable_heuristic_disc_switching"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --initial_disc_type %INITIAL_DISC_TYPE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_switch_check_interval %DISC_SWITCH_CHECK_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_switch_min_steps_between %DISC_SWITCH_MIN_STEPS_BETWEEN%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_switch_problem_state_count_thresh %DISC_SWITCH_PROBLEM_STATE_COUNT_THRESH%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --video_global_wubu_d_input_tangent_dim %VIDEO_GLOBAL_WUBU_D_INPUT_TANGENT_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --video_global_wubu_d_output_feature_dim %VIDEO_GLOBAL_WUBU_D_OUTPUT_FEATURE_DIM%"
IF /I "%DISC_USE_GLOBAL_STATS_AUX_VIDEO_GLOBAL_WUBU%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_use_global_stats_aux_video_global_wubu"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_global_stats_mlp_hidden_dim_video_global_wubu %DISC_GLOBAL_STATS_MLP_HIDDEN_DIM_VIDEO_GLOBAL_WUBU%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_dropout %WUBU_DROPOUT%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_num_levels %WUBU_S_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_hyperbolic_dims %WUBU_S_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_initial_curvatures %WUBU_S_INITIAL_CURVATURES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_initial_scales %WUBU_S_INITIAL_SCALES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_initial_spread_values %WUBU_S_INITIAL_SPREAD_VALUES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_boundary_points_per_level %WUBU_S_BOUNDARY_POINTS_PER_LEVEL%"
IF /I "%WUBU_S_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_use_rotation"
IF /I "%WUBU_S_PHI_INFLUENCE_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_curvature"
IF /I "%WUBU_S_PHI_INFLUENCE_ROTATION_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_rotation_init"

IF /I "%USE_WUBU_MOTION_BRANCH%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_wubu_motion_branch"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gaad_motion_num_regions %GAAD_MOTION_NUM_REGIONS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gaad_motion_decomposition_type %GAAD_MOTION_DECOMP_TYPE%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_num_levels %WUBU_M_NUM_LEVELS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_hyperbolic_dims %WUBU_M_HYPERBOLIC_DIMS%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_initial_curvatures %WUBU_M_INITIAL_CURVATURES%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_initial_scales %WUBU_M_INITIAL_SCALES%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_initial_spread_values %WUBU_M_INITIAL_SPREAD_VALUES%"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_boundary_points_per_level %WUBU_M_BOUNDARY_POINTS_PER_LEVEL%"
    IF /I "%WUBU_M_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_use_rotation"
    IF /I "%WUBU_M_PHI_INFLUENCE_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_phi_influence_curvature"
    IF /I "%WUBU_M_PHI_INFLUENCE_ROTATION_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_m_phi_influence_rotation_init"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optical_flow_net_type %OPTICAL_FLOW_NET_TYPE%"
    IF /I "%FREEZE_FLOW_NET%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --freeze_flow_net"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --flow_stats_components %FLOW_STATS_COMPONENTS%"
)

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_num_levels %WUBU_T_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_hyperbolic_dims %WUBU_T_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_initial_curvatures %WUBU_T_INITIAL_CURVATURES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_initial_scales %WUBU_T_INITIAL_SCALES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_initial_spread_values %WUBU_T_INITIAL_SPREAD_VALUES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_boundary_points_per_level %WUBU_T_BOUNDARY_POINTS_PER_LEVEL%"
IF /I "%WUBU_T_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_use_rotation"
IF /I "%WUBU_T_PHI_INFLUENCE_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_phi_influence_curvature"
IF /I "%WUBU_T_PHI_INFLUENCE_ROTATION_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_t_phi_influence_rotation_init"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_global_video_num_levels %WUBU_D_GLOBAL_VIDEO_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_global_video_hyperbolic_dims %WUBU_D_GLOBAL_VIDEO_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_global_video_initial_curvatures %WUBU_D_GLOBAL_VIDEO_INITIAL_CURVATURES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_global_video_initial_scales %WUBU_D_GLOBAL_VIDEO_INITIAL_SCALES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_global_video_initial_spread_values %WUBU_D_GLOBAL_VIDEO_INITIAL_SPREAD_VALUES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_d_global_video_boundary_points_per_level %WUBU_D_GLOBAL_VIDEO_BOUNDARY_POINTS_PER_LEVEL%"
IF /I "%WUBU_D_GLOBAL_VIDEO_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_global_video_use_rotation"
IF /I "%WUBU_D_GLOBAL_VIDEO_PHI_INFLUENCE_CURVATURE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_global_video_phi_influence_curvature"
IF /I "%WUBU_D_GLOBAL_VIDEO_PHI_INFLUENCE_ROTATION_INIT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_global_video_phi_influence_rotation_init"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --epochs %EPOCHS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --batch_size %BATCH_SIZE_PER_GPU%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --grad_accum_steps %GRAD_ACCUM_STEPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --learning_rate_gen %LEARNING_RATE_GEN%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --learning_rate_disc %LEARNING_RATE_DISC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --learning_rate_disc_alt %LEARNING_RATE_DISC_ALT%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --risgd_max_grad_norm %RISGD_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --global_max_grad_norm %GLOBAL_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --seed %TRAIN_SEED%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_workers %TRAIN_NUM_WORKERS%"
IF /I "%USE_AMP%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_amp"
IF /I "%DETECT_ANOMALY%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --detect_anomaly"
IF /I "%LOAD_STRICT%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_strict"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_recon %LAMBDA_RECON%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_recon_dft %LAMBDA_RECON_DFT%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_recon_dct %LAMBDA_RECON_DCT%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_kl %LAMBDA_KL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_gan %LAMBDA_GAN%"

IF /I "%Q_CONTROLLER_ENABLED%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled"
IF /I "%RESET_Q_CONTROLLERS_ON_LOAD%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --reset_q_controllers_on_load"
IF /I "%RESET_LKL_Q_CONTROLLER_ON_LOAD%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --reset_lkl_q_controller_on_load"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_kl_update_interval %LAMBDA_KL_UPDATE_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --min_lambda_kl_q_control %MIN_LAMBDA_KL_Q_CONTROL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --max_lambda_kl_q_control %MAX_LAMBDA_KL_Q_CONTROL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_lkl_scale_options %Q_LKL_SCALE_OPTIONS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_lkl_lr_mom_probation_steps %Q_LKL_LR_MOM_PROBATION_STEPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_lkl_action_probation_steps %Q_LKL_ACTION_PROBATION_STEPS%"

IF /I "%ENABLE_HEURISTIC_INTERVENTIONS%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --enable_heuristic_interventions"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_check_interval %HEURISTIC_CHECK_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_short_term_history_len %HEURISTIC_SHORT_TERM_HISTORY_LEN%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_trigger_count_thresh %HEURISTIC_TRIGGER_COUNT_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_d_strong_thresh %HEURISTIC_D_STRONG_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_d_weak_thresh %HEURISTIC_D_WEAK_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_d_very_weak_thresh %HEURISTIC_D_VERY_WEAK_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_g_stalled_thresh %HEURISTIC_G_STALLED_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_g_winning_thresh %HEURISTIC_G_WINNING_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_g_very_much_winning_thresh %HEURISTIC_G_VERY_MUCH_WINNING_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_kl_high_thresh %HEURISTIC_KL_HIGH_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_recon_stagnation_improvement_thresh_rel %HEURISTIC_RECON_STAGNATION_IMPROVEMENT_THRESH_REL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --target_good_recon_thresh_heuristic_video %TARGET_GOOD_RECON_THRESH_HEURISTIC_VIDEO%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_q_reward_stagnation_thresh %HEURISTIC_Q_REWARD_STAGNATION_THRESH%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_recon_boost_factor_video %HEURISTIC_RECON_BOOST_FACTOR_VIDEO%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_feat_match_heuristic_video %LAMBDA_FEAT_MATCH_HEURISTIC_VIDEO%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_g_easy_win_penalty_heuristic_video %LAMBDA_G_EASY_WIN_PENALTY_HEURISTIC_VIDEO%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --g_easy_win_penalty_eps_denom %G_EASY_WIN_PENALTY_EPS_DENOM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --max_g_easy_win_penalty_abs %MAX_G_EASY_WIN_PENALTY_ABS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_active_d_lr_boost_factor %HEURISTIC_ACTIVE_D_LR_BOOST_FACTOR%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_d_q_explore_boost_epsilon %HEURISTIC_D_Q_EXPLORE_BOOST_EPSILON%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_d_q_explore_duration %HEURISTIC_D_Q_EXPLORE_DURATION%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_min_lambda_gan_factor %HEURISTIC_MIN_LAMBDA_GAN_FACTOR%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --heuristic_max_lambda_gan_factor %HEURISTIC_MAX_LAMBDA_GAN_FACTOR%"
IF DEFINED FORCE_START_EPOCH_ON_LOAD (
    IF NOT "%FORCE_START_EPOCH_ON_LOAD%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --force_start_epoch_on_load %FORCE_START_EPOCH_ON_LOAD%"
)
IF DEFINED FORCE_START_GSTEP_ON_LOAD (
    IF NOT "%FORCE_START_GSTEP_ON_LOAD%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --force_start_gstep_on_load %FORCE_START_GSTEP_ON_LOAD%"
)

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --log_interval %LOG_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --save_interval %SAVE_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --save_epoch_interval %SAVE_EPOCH_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_interval_epochs %VALIDATION_INTERVAL_EPOCHS%"
IF /I "%DISABLE_VAL_TQDM%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disable_val_tqdm"
IF /I "%USE_LPIPS_FOR_VERIFICATION%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_lpips_for_verification"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --validation_split_fraction %VALIDATION_SPLIT_FRACTION%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_block_size %VAL_BLOCK_SIZE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_primary_metric %VAL_PRIMARY_METRIC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_val_samples_to_log %NUM_VAL_SAMPLES_TO_LOG%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --demo_num_samples %DEMO_NUM_SAMPLES%"

IF /I "%WANDB_ENABLED%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb --wandb_project %WANDB_PROJECT%"
    IF DEFINED WANDB_RUN_NAME (
        IF NOT "%WANDB_RUN_NAME%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_run_name "%WANDB_RUN_NAME%""
    )
)
IF %WANDB_LOG_TRAIN_RECON_INTERVAL% GTR 0 SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_log_train_recon_interval %WANDB_LOG_TRAIN_RECON_INTERVAL%"
IF %WANDB_LOG_FIXED_NOISE_SAMPLES_INTERVAL% GTR 0 SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_log_fixed_noise_samples_interval %WANDB_LOG_FIXED_NOISE_SAMPLES_INTERVAL%"


REM =====================================================================
REM Pre-Run Echo
REM =====================================================================
ECHO ======================================================
ECHO WuBuGAADHybridGen VAE-GAN DFT+DCT (v0.3) - Comprehensive Run
ECHO Python: %PYTHON_EXE%
ECHO Script Name: %SCRIPT_NAME%
ECHO Checkpoints: %CHECKPOINT_OUTPUT_DIR%
ECHO DFT Appearance: %USE_DFT_FEATURES_APPEARANCE%
ECHO DCT Appearance: %USE_DCT_FEATURES_APPEARANCE%
ECHO Motion Branch: %USE_WUBU_MOTION_BRANCH% (Optical Flow: %OPTICAL_FLOW_NET_TYPE%)
ECHO Discriminator Variants (Primary/Alt): %PRIMARY_DISC_ARCHITECTURE_VARIANT% / %ALT_DISC_ARCHITECTURE_VARIANT%
ECHO Heuristic D Switching: %ENABLE_HEURISTIC_DISC_SWITCHING% (Initial Pref: %INITIAL_DISC_TYPE%)
ECHO Heuristic Interventions: %ENABLE_HEURISTIC_INTERVENTIONS%
ECHO AMP: %USE_AMP%
ECHO Q-Controller: %Q_CONTROLLER_ENABLED% (Lambda_KL Update Interval: %LAMBDA_KL_UPDATE_INTERVAL%)
ECHO WANDB: %WANDB_ENABLED% (Project: %WANDB_PROJECT%)
ECHO Learning Rate (Gen/Disc/AltDisc): %LEARNING_RATE_GEN% / %LEARNING_RATE_DISC% / %LEARNING_RATE_DISC_ALT%
ECHO Batch Size (Global/PerGPU): %GLOBAL_BATCH_SIZE% / %BATCH_SIZE_PER_GPU%
ECHO Grad Accum Steps: %GRAD_ACCUM_STEPS%
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