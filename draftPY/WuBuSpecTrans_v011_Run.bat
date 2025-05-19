@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM ############################################################################
REM # SECTION: Project Paths & Python Environment
REM ############################################################################
SET "PROJECT_ROOT=%~dp0.."
SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT_NAME=WuBuSpecTrans_v0.1.1.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"
SET "DATA_DIR_BASE=%PROJECT_ROOT%\data"
SET "CHECKPOINT_OUTPUT_DIR_BASE=%PROJECT_ROOT%\checkpoints"

IF EXIST "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
) ELSE IF EXIST "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) ELSE (
    SET "PYTHON_EXE=python"
)

REM ############################################################################
REM # SECTION: Run Configuration & Checkpoint Management
REM ############################################################################
SET "RUN_NAME_SUFFIX=Run_MelReconFocus_VRAMPush"
SET "CHECKPOINT_OUTPUT_DIR=%CHECKPOINT_OUTPUT_DIR_BASE%\WuBuSpecTrans_v011_%RUN_NAME_SUFFIX%"
SET "LOAD_CHECKPOINT_VALUE="
SET "BEST_CKPT_NAME=wubuspectrans_ckpt_v011_best.pt"
SET "LATEST_EPOCH_CKPT_NAME="

IF EXIST "%CHECKPOINT_OUTPUT_DIR%\%BEST_CKPT_NAME%" (
    SET "LOAD_CHECKPOINT_VALUE=%CHECKPOINT_OUTPUT_DIR%\%BEST_CKPT_NAME%"
) ELSE (
    FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "%CHECKPOINT_OUTPUT_DIR%\wubuspectrans_ckpt_v011_ep*_step*.pt"') DO (
        SET "LATEST_EPOCH_CKPT_NAME=%%F"
        GOTO FoundLatestEpochCkpt_ReconFocus
    )
    :FoundLatestEpochCkpt_ReconFocus
    IF DEFINED LATEST_EPOCH_CKPT_NAME (
        SET "LOAD_CHECKPOINT_VALUE=%CHECKPOINT_OUTPUT_DIR%\!LATEST_EPOCH_CKPT_NAME!"
    )
)

REM ############################################################################
REM # SECTION: Core Training & Model Hyperparameters (Max Detail & Params)
REM ############################################################################
SET "EPOCHS=2500"
SET "GLOBAL_BATCH_SIZE=32"
SET "NPROC_PER_NODE=1"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
SET "LEARNING_RATE_GEN=5e-5"
SET "LEARNING_RATE_DISC=3e-5"
SET "LEARNING_RATE_DISC_ALT=3e-5"
SET "RISGD_MAX_GRAD_NORM=2.5"
SET "GLOBAL_MAX_GRAD_NORM=3.5"

SET "LATENT_DIM=768"
SET "ENCODER_INITIAL_TANGENT_DIM=256"

REM ############################################################################
REM # SECTION: Loss Weights (Still Focusing Mel Reconstruction Initially)
REM ############################################################################
SET "LAMBDA_RECON=1.5"
SET "LAMBDA_KL=9e-3"
SET "LAMBDA_GAN=0.3"

REM ############################################################################
REM # SECTION: Audio Processing & Dataset (Pushing Detail)
REM ############################################################################
SET "AUDIO_DATA_PATH=%DATA_DIR_BASE%\demo_audio_data_dir"
SET "VALIDATION_AUDIO_PATH="
SET "SAMPLE_RATE=44100"
SET "N_FFT=2048"
SET "HOP_LENGTH=256"
SET "N_MELS=256"
SET "FMIN=20.0"
SET "FMAX="
SET "SEGMENT_DURATION_SEC=1.0"
SET "SEGMENT_OVERLAP_SEC=0.0"
SET "DB_NORM_MIN=-90.0"
SET "DB_NORM_MAX=0.0"
SET "PRELOAD_AUDIO_DATASET_TO_RAM=true"
SET "VALIDATION_SPLIT_FRACTION=0.1"

REM ############################################################################
REM # SECTION: GAAD & DCT Processing (Max Detail Configuration)
REM ############################################################################
SET "GAAD_NUM_REGIONS=192"
SET "GAAD_DECOMP_TYPE=hybrid"
SET "GAAD_MIN_SIZE_PX=3"
SET "REGION_PROC_SIZE_T=32"
SET "REGION_PROC_SIZE_F=32"
SET "DCT_NORM_TYPE=tanh"
SET "DCT_NORM_GLOBAL_SCALE=150.0"
SET "DCT_NORM_TANH_SCALE=50.0"

REM ############################################################################
REM # SECTION: Discriminator Configuration (Primary Mel D might need more capacity)
REM ############################################################################
SET "DISCRIMINATOR_INPUT_TYPE=mel"
SET "DISC_APPLY_SPECTRAL_NORM=true"
SET "DISC_BASE_DISC_CHANNELS=96"
SET "DISC_MAX_DISC_CHANNELS=768"
SET "DISC_TARGET_FINAL_FEATURE_DIM=4"

REM ############################################################################
REM # SECTION: WuBu Stack Configurations (Increased Capacity for Max Detail)
REM ############################################################################
SET "WUBU_DROPOUT=0.05"

SET "WUBU_S_NUM_LEVELS=4"
SET "WUBU_S_HYPERBOLIC_DIMS=256 192 128 96"
SET "WUBU_S_INITIAL_CURVATURES=0.9 0.7 0.6 0.5"
SET "WUBU_S_USE_ROTATION=false"
SET "WUBU_S_PHI_CURVATURE=true"
SET "WUBU_S_PHI_ROT_INIT=false"
SET "WUBU_S_OUTPUT_DIM_ENCODER=512"

SET "WUBU_G_NUM_LEVELS=4"
SET "WUBU_G_HYPERBOLIC_DIMS=256 384 512 1024"
SET "WUBU_G_INITIAL_CURVATURES=0.5 0.6 0.7 0.9"
SET "WUBU_G_USE_ROTATION=false"
SET "WUBU_G_PHI_CURVATURE=true"
SET "WUBU_G_PHI_ROT_INIT=false"

SET "WUBU_D_NUM_LEVELS=3"
SET "WUBU_D_HYPERBOLIC_DIMS=256 192 128"
SET "WUBU_D_INITIAL_CURVATURES=0.8 0.6 0.5"
SET "WUBU_D_USE_ROTATION=false"
SET "WUBU_D_PHI_CURVATURE=true"
SET "WUBU_D_PHI_ROT_INIT=false"
SET "WUBU_D_OUTPUT_DIM=128"

REM ############################################################################
REM # SECTION: Q-Learning & Heuristics (Initially Subdued for VAE Lambdas)
REM ############################################################################
SET "Q_CONTROLLER_ENABLED=true"
SET "RESET_Q_CONTROLLERS_ON_LOAD=false"
SET "RESET_LKL_Q_CONTROLLER_ON_LOAD=true"

SET "LAMBDA_KL_UPDATE_INTERVAL=5"
SET "MIN_LAMBDA_KL_Q_CONTROL=1e-6"
SET "MAX_LAMBDA_KL_Q_CONTROL=0.05"
SET "Q_LKL_SCALE_OPTIONS=0.90 0.95 1.0 1.05 1.10"
SET "Q_LKL_LR_MOM_PROBATION_STEPS=1"
SET "Q_LKL_ACTION_PROBATION_STEPS=1"

SET "ENABLE_HEURISTIC_INTERVENTIONS=true"
SET "ENABLE_HEURISTIC_DISC_SWITCHING=true"
SET "INITIAL_DISC_TYPE=%DISCRIMINATOR_INPUT_TYPE%"
SET "HEURISTIC_CHECK_INTERVAL=10"
SET "HEURISTIC_SHORT_TERM_HISTORY_LEN=7"
SET "HEURISTIC_TRIGGER_COUNT_THRESH=2"

SET "DISC_SWITCH_CHECK_INTERVAL=10"
SET "DISC_SWITCH_MIN_STEPS_BETWEEN=200"
SET "DISC_SWITCH_PROBLEM_STATE_COUNT_THRESH=3"

SET "HEURISTIC_D_STRONG_THRESH=0.25"
SET "HEURISTIC_D_WEAK_THRESH=1.0"
SET "HEURISTIC_D_VERY_WEAK_THRESH=1.8"
SET "HEURISTIC_G_STALLED_THRESH=1.5"
SET "HEURISTIC_G_WINNING_THRESH=0.2"
SET "HEURISTIC_G_VERY_MUCH_WINNING_THRESH=0.05"
SET "HEURISTIC_KL_HIGH_THRESH=25.0"
SET "HEURISTIC_RECON_STAGNATION_IMPROVEMENT_THRESH_REL=0.001"
SET "TARGET_GOOD_RECON_THRESH_HEURISTIC=0.03"
SET "HEURISTIC_Q_REWARD_STAGNATION_THRESH=-0.25"

SET "HEURISTIC_RECON_BOOST_FACTOR=1.0"
SET "LAMBDA_FEAT_MATCH_HEURISTIC=0.75"
SET "LAMBDA_G_EASY_WIN_PENALTY_HEURISTIC=1.5"
SET "HEURISTIC_ACTIVE_D_LR_BOOST_FACTOR=1.8"
SET "HEURISTIC_D_Q_EXPLORE_BOOST_EPSILON=0.7"
SET "HEURISTIC_D_Q_EXPLORE_DURATION=10"

REM ############################################################################
REM # SECTION: General Training Settings & Logging
REM ############################################################################
SET "TRAIN_SEED=42"
SET "TRAIN_NUM_WORKERS=4"
SET "USE_AMP=true"
SET "DETECT_ANOMALY=false"
SET "LOAD_STRICT=true"
SET "DDP_FIND_UNUSED_PARAMS_D=false"
SET "FORCE_START_EPOCH_ON_LOAD="
SET "FORCE_START_GSTEP_ON_LOAD="
SET "DISABLE_VAL_TQDM=false"

SET "LOG_INTERVAL=1"
SET "SAVE_INTERVAL=5000"
SET "SAVE_EPOCH_INTERVAL=1"
SET "VALIDATION_INTERVAL_EPOCHS=1"
SET "USE_LPIPS_FOR_MEL_VERIFICATION=true"
SET "VAL_PRIMARY_METRIC=avg_val_lpips_mel"
SET "NUM_VAL_SAMPLES_TO_LOG=4"
SET "DEMO_NUM_SAMPLES=5"

SET "WANDB_ENABLED=true"
SET "WANDB_PROJECT=WuBuSpecTransV011_ReconFocus"
SET "WANDB_RUN_NAME=%RUN_NAME_SUFFIX%_Lat%LATENT_DIM%_SR%SAMPLE_RATE%"
SET "WANDB_LOG_TRAIN_RECON_INTERVAL=10"
SET "TRAIN_TARGET_LOG_FREQ_MULTIPLIER=2"
SET "WANDB_LOG_FIXED_NOISE_INTERVAL=25"

REM ############################################################################
REM # SECTION: Construct Script Arguments
REM ############################################################################
SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --audio_dir_path "%AUDIO_DATA_PATH%""
IF DEFINED VALIDATION_AUDIO_PATH ( IF NOT "!VALIDATION_AUDIO_PATH!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --validation_audio_dir_path "%VALIDATION_AUDIO_PATH%"" )
IF DEFINED LOAD_CHECKPOINT_VALUE ( IF NOT "!LOAD_CHECKPOINT_VALUE!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_checkpoint "%LOAD_CHECKPOINT_VALUE%"" )
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR%""
IF /I "!LOAD_STRICT!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_strict"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --seed %TRAIN_SEED%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --num_workers %TRAIN_NUM_WORKERS%"
IF /I "!USE_AMP!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_amp"
IF /I "!DETECT_ANOMALY!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --detect_anomaly"
IF /I "!DDP_FIND_UNUSED_PARAMS_D!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --ddp_find_unused_params_d"

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --epochs %EPOCHS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --batch_size %BATCH_SIZE_PER_GPU%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --grad_accum_steps %GRAD_ACCUM_STEPS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --learning_rate_gen %LEARNING_RATE_GEN%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --learning_rate_disc %LEARNING_RATE_DISC%"
IF DEFINED LEARNING_RATE_DISC_ALT ( IF NOT "!LEARNING_RATE_DISC_ALT!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --learning_rate_disc_alt %LEARNING_RATE_DISC_ALT%" )
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --risgd_max_grad_norm %RISGD_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --global_max_grad_norm %GLOBAL_MAX_GRAD_NORM%"

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_recon %LAMBDA_RECON%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_kl %LAMBDA_KL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_gan %LAMBDA_GAN%"

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --sample_rate %SAMPLE_RATE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --n_fft %N_FFT%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --hop_length %HOP_LENGTH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --n_mels %N_MELS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --fmin %FMIN%"
IF DEFINED FMAX ( IF NOT "!FMAX!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --fmax %FMAX%" )
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --segment_duration_sec %SEGMENT_DURATION_SEC%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --segment_overlap_sec %SEGMENT_OVERLAP_SEC%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --db_norm_min %DB_NORM_MIN%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --db_norm_max %DB_NORM_MAX%"
IF /I "!PRELOAD_AUDIO_DATASET_TO_RAM!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --preload_audio_dataset_to_ram"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --validation_split_fraction %VALIDATION_SPLIT_FRACTION%"

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gaad_num_regions %GAAD_NUM_REGIONS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gaad_decomposition_type %GAAD_DECOMP_TYPE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --gaad_min_size_px %GAAD_MIN_SIZE_PX%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --region_proc_size_t %REGION_PROC_SIZE_T%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --region_proc_size_f %REGION_PROC_SIZE_F%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dct_norm_type %DCT_NORM_TYPE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dct_norm_global_scale %DCT_NORM_GLOBAL_SCALE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dct_norm_tanh_scale %DCT_NORM_TANH_SCALE%"

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --latent_dim %LATENT_DIM%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --encoder_initial_tangent_dim %ENCODER_INITIAL_TANGENT_DIM%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_input_type %DISCRIMINATOR_INPUT_TYPE%"
IF /I "!DISC_APPLY_SPECTRAL_NORM!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_apply_spectral_norm"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_base_disc_channels %DISC_BASE_DISC_CHANNELS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_max_disc_channels %DISC_MAX_DISC_CHANNELS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_target_final_feature_dim %DISC_TARGET_FINAL_FEATURE_DIM%"

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_dropout %WUBU_DROPOUT%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_num_levels %WUBU_S_NUM_LEVELS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_hyperbolic_dims %WUBU_S_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_initial_curvatures %WUBU_S_INITIAL_CURVATURES%"
IF /I "!WUBU_S_USE_ROTATION!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_use_rotation"
IF /I "!WUBU_S_PHI_CURVATURE!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_curvature"
IF /I "!WUBU_S_PHI_ROT_INIT!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_phi_influence_rotation_init"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_s_output_dim_encoder %WUBU_S_OUTPUT_DIM_ENCODER%"

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_num_levels %WUBU_G_NUM_LEVELS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_hyperbolic_dims %WUBU_G_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_initial_curvatures %WUBU_G_INITIAL_CURVATURES%"
IF /I "!WUBU_G_USE_ROTATION!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_use_rotation"
IF /I "!WUBU_G_PHI_CURVATURE!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_phi_influence_curvature"
IF /I "!WUBU_G_PHI_ROT_INIT!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_g_phi_influence_rotation_init"

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_num_levels %WUBU_D_NUM_LEVELS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_hyperbolic_dims %WUBU_D_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_initial_curvatures %WUBU_D_INITIAL_CURVATURES%"
IF /I "!WUBU_D_USE_ROTATION!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_use_rotation"
IF /I "!WUBU_D_PHI_CURVATURE!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_phi_influence_curvature"
IF /I "!WUBU_D_PHI_ROT_INIT!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_phi_influence_rotation_init"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_d_output_dim %WUBU_D_OUTPUT_DIM%"

IF /I "!Q_CONTROLLER_ENABLED!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled"
IF /I "!RESET_Q_CONTROLLERS_ON_LOAD!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --reset_q_controllers_on_load"
IF /I "!RESET_LKL_Q_CONTROLLER_ON_LOAD!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --reset_lkl_q_controller_on_load"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_kl_update_interval %LAMBDA_KL_UPDATE_INTERVAL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --min_lambda_kl_q_control %MIN_LAMBDA_KL_Q_CONTROL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --max_lambda_kl_q_control %MAX_LAMBDA_KL_Q_CONTROL%"
IF DEFINED Q_LKL_SCALE_OPTIONS ( IF NOT "!Q_LKL_SCALE_OPTIONS!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_lkl_scale_options %Q_LKL_SCALE_OPTIONS%" )
IF DEFINED Q_LKL_LR_MOM_PROBATION_STEPS ( IF NOT "!Q_LKL_LR_MOM_PROBATION_STEPS!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_lkl_lr_mom_probation_steps %Q_LKL_LR_MOM_PROBATION_STEPS%" )
IF DEFINED Q_LKL_ACTION_PROBATION_STEPS ( IF NOT "!Q_LKL_ACTION_PROBATION_STEPS!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_lkl_action_probation_steps %Q_LKL_ACTION_PROBATION_STEPS%" )

IF /I "!ENABLE_HEURISTIC_INTERVENTIONS!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --enable_heuristic_interventions"
IF /I "!ENABLE_HEURISTIC_DISC_SWITCHING!"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --enable_heuristic_disc_switching"
    IF DEFINED INITIAL_DISC_TYPE ( IF NOT "!INITIAL_DISC_TYPE!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --initial_disc_type %INITIAL_DISC_TYPE%" )
)
IF DEFINED HEURISTIC_CHECK_INTERVAL ( IF NOT "!HEURISTIC_CHECK_INTERVAL!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_check_interval %HEURISTIC_CHECK_INTERVAL%" ) ELSE ( SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_check_interval %LOG_INTERVAL%" )
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_short_term_history_len %HEURISTIC_SHORT_TERM_HISTORY_LEN%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_trigger_count_thresh %HEURISTIC_TRIGGER_COUNT_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_switch_check_interval %DISC_SWITCH_CHECK_INTERVAL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_switch_min_steps_between %DISC_SWITCH_MIN_STEPS_BETWEEN%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_switch_problem_state_count_thresh %DISC_SWITCH_PROBLEM_STATE_COUNT_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_d_strong_thresh %HEURISTIC_D_STRONG_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_d_weak_thresh %HEURISTIC_D_WEAK_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_d_very_weak_thresh %HEURISTIC_D_VERY_WEAK_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_g_stalled_thresh %HEURISTIC_G_STALLED_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_g_winning_thresh %HEURISTIC_G_WINNING_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_g_very_much_winning_thresh %HEURISTIC_G_VERY_MUCH_WINNING_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_kl_high_thresh %HEURISTIC_KL_HIGH_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_recon_stagnation_improvement_thresh_rel %HEURISTIC_RECON_STAGNATION_IMPROVEMENT_THRESH_REL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --target_good_recon_thresh_heuristic %TARGET_GOOD_RECON_THRESH_HEURISTIC%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_q_reward_stagnation_thresh %HEURISTIC_Q_REWARD_STAGNATION_THRESH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_recon_boost_factor %HEURISTIC_RECON_BOOST_FACTOR%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_feat_match_heuristic %LAMBDA_FEAT_MATCH_HEURISTIC%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_g_easy_win_penalty_heuristic %LAMBDA_G_EASY_WIN_PENALTY_HEURISTIC%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_active_d_lr_boost_factor %HEURISTIC_ACTIVE_D_LR_BOOST_FACTOR%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_d_q_explore_boost_epsilon %HEURISTIC_D_Q_EXPLORE_BOOST_EPSILON%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --heuristic_d_q_explore_duration %HEURISTIC_D_Q_EXPLORE_DURATION%"
IF DEFINED FORCE_START_EPOCH_ON_LOAD ( IF NOT "!FORCE_START_EPOCH_ON_LOAD!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --force_start_epoch_on_load %FORCE_START_EPOCH_ON_LOAD%" )
IF DEFINED FORCE_START_GSTEP_ON_LOAD ( IF NOT "!FORCE_START_GSTEP_ON_LOAD!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --force_start_gstep_on_load %FORCE_START_GSTEP_ON_LOAD%" )

SET "SCRIPT_ARGS=!SCRIPT_ARGS! --log_interval %LOG_INTERVAL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --save_interval %SAVE_INTERVAL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --save_epoch_interval %SAVE_EPOCH_INTERVAL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --validation_interval_epochs %VALIDATION_INTERVAL_EPOCHS%"
IF /I "!DISABLE_VAL_TQDM!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disable_val_tqdm"
IF /I "!USE_LPIPS_FOR_MEL_VERIFICATION!"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_lpips_for_mel_verification"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --val_primary_metric %VAL_PRIMARY_METRIC%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --num_val_samples_to_log %NUM_VAL_SAMPLES_TO_LOG%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --demo_num_samples %DEMO_NUM_SAMPLES%"

IF /I "!WANDB_ENABLED!"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb --wandb_project %WANDB_PROJECT%"
    IF DEFINED WANDB_RUN_NAME ( IF NOT "!WANDB_RUN_NAME!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_run_name "%WANDB_RUN_NAME%"" )
)
IF %WANDB_LOG_TRAIN_RECON_INTERVAL% GTR 0 SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_log_train_recon_interval %WANDB_LOG_TRAIN_RECON_INTERVAL%"
IF DEFINED TRAIN_TARGET_LOG_FREQ_MULTIPLIER ( IF NOT "!TRAIN_TARGET_LOG_FREQ_MULTIPLIER!"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --train_target_log_freq_multiplier %TRAIN_TARGET_LOG_FREQ_MULTIPLIER%" )
IF %WANDB_LOG_FIXED_NOISE_INTERVAL% GTR 0 SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_log_fixed_noise_samples_interval %WANDB_LOG_FIXED_NOISE_INTERVAL%"

REM ############################################################################
REM # SECTION: Display Configuration Summary
REM ############################################################################
ECHO ======================================================
ECHO WuBuSpecTrans VAE-GAN - Audio Run (%RUN_NAME_SUFFIX%)
ECHO Python Executable: %PYTHON_EXE%
ECHO Python Script: %SCRIPT_NAME%
ECHO Checkpoint Output Directory: %CHECKPOINT_OUTPUT_DIR%
ECHO Loading Checkpoint Value: %LOAD_CHECKPOINT_VALUE%
ECHO Audio Data Source: %AUDIO_DATA_PATH%
IF DEFINED VALIDATION_AUDIO_PATH ( IF NOT "%VALIDATION_AUDIO_PATH%"=="" ECHO Validation Audio Source: %VALIDATION_AUDIO_PATH% ) ELSE ( ECHO Validation from Training Split: %VALIDATION_SPLIT_FRACTION% )
ECHO Initial Discriminator Type (Arg): %DISCRIMINATOR_INPUT_TYPE%
ECHO Heuristic D Switching Enabled: %ENABLE_HEURISTIC_DISC_SWITCHING%
IF /I "%ENABLE_HEURISTIC_DISC_SWITCHING%"=="true" ECHO Initial Active D Type for Switching Logic: %INITIAL_DISC_TYPE%
ECHO AMP Enabled: %USE_AMP%
ECHO Q-Controller Enabled: %Q_CONTROLLER_ENABLED%
ECHO Reset Q-Controllers on Load: %RESET_Q_CONTROLLERS_ON_LOAD%
ECHO Reset LKL Q-Controller on Load: %RESET_LKL_Q_CONTROLLER_ON_LOAD%
ECHO Advanced Heuristics Enabled: %ENABLE_HEURISTIC_INTERVENTIONS%
ECHO Batch Size per GPU: %BATCH_SIZE_PER_GPU% (Global: %GLOBAL_BATCH_SIZE% / NPROC: %NPROC_PER_NODE%)
ECHO ======================================================
ECHO.

REM ############################################################################
REM # SECTION: Environment Activation & Script Execution
REM ############################################################################
IF NOT EXIST "%PYTHON_EXE%" (
    ECHO ERROR: Python executable not found at %PYTHON_EXE%
    GOTO :EndScriptReconFocus
)
SET "VENV_ACTIVATE_PATH="
FOR %%F IN ("%PYTHON_EXE%") DO SET "VENV_ACTIVATE_PATH=%%~dpFactivate.bat"
IF EXIST "%VENV_ACTIVATE_PATH%" (
    ECHO Activating virtual environment: %VENV_ACTIVATE_PATH%
    CALL "%VENV_ACTIVATE_PATH%"
    IF ERRORLEVEL 1 ( ECHO WARNING: Failed to activate venv, proceeding with system Python. )
) ELSE (
    ECHO No venv activate script found at expected path. Using system Python or current environment.
)
ECHO.

IF NOT EXIST "%CHECKPOINT_OUTPUT_DIR%" MKDIR "%CHECKPOINT_OUTPUT_DIR%"
IF NOT EXIST "%DATA_DIR_BASE%" MKDIR "%DATA_DIR_BASE%"
IF NOT EXIST "%AUDIO_DATA_PATH%" MKDIR "%AUDIO_DATA_PATH%"

ECHO Starting training script: %SCRIPT_NAME%
ECHO.
ECHO Full command to be executed:
ECHO "%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
ECHO.

IF %NPROC_PER_NODE% EQU 1 (
    "%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
) ELSE (
    ECHO Launching with torch.distributed.run for %NPROC_PER_NODE% processes.
    "%PYTHON_EXE%" -m torch.distributed.run --nproc_per_node=%NPROC_PER_NODE% --master_port=29522 "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
)

SET "EXIT_CODE=%ERRORLEVEL%"
ECHO.
IF %EXIT_CODE% NEQ 0 (
    ECHO **************************************
    ECHO * SCRIPT FAILED with exit code %EXIT_CODE% *
    ECHO **************************************
) ELSE (
    ECHO ****************************************
    ECHO * SCRIPT FINISHED successfully *
    ECHO ****************************************
)

:EndScriptReconFocus
IF DEFINED PROMPT_AFTER_RUN (
    IF /I "%PROMPT_AFTER_RUN%"=="true" PAUSE
) ELSE (
    TIMEOUT /T 15 /NOBREAK >nul
)
ENDLOCAL