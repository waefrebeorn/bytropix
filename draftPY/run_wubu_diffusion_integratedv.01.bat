@echo OFF
REM ============================================================================
REM Batch file for FUNCTIONAL DEMO of WuBuNestDiffusion_v0.04_GAAD_Phi_Infused.py
REM V0.04 - Updated for new script arguments, video data input, GAAD, and Phi features.
REM         Defaults to using the script's dummy video generation for easy first run.
REM ============================================================================
SETLOCAL ENABLEDELAYEDEXPANSION

REM --- Project and Script Paths ---
SET "PROJECT_ROOT=%~dp0.."
REM Adjust PYTHON_EXE if your venv is named differently or located elsewhere
IF EXIST "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
) ELSE IF EXIST "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) ELSE (
    ECHO Python executable not found in typical venv paths. Set PYTHON_EXE manually.
    SET "PYTHON_EXE=python" REM Fallback, hoping python is in PATH
)
SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT_NAME=WuBuNestDiffusion_v0.04_GAAD_Phi_Infused.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

REM --- DDP Settings (NPROC_PER_NODE=1 for single GPU demo) ---
SET "NPROC_PER_NODE=1"
SET "MASTER_ADDR=localhost"
SET "MASTER_PORT=29504" REM Changed port slightly from old script

REM --- Data and Output Directories ---
SET "DATA_DIR_BASE=%PROJECT_ROOT%\data"
SET "CHECKPOINT_OUTPUT_DIR=%PROJECT_ROOT%\checkpoints\WuBuGAADPhi_FunctionalDemo_v04"

REM For initial demo, script will create dummy video if this path is used and empty
SET "VIDEO_DATA_PATH=%DATA_DIR_BASE%\demo_video_data_dir"
REM To use your own data, change VIDEO_DATA_PATH to your video folder/file
REM SET "VIDEO_DATA_PATH=%DATA_DIR_BASE%\my_actual_videos"

REM --- Model & Training Hyperparameters for DEMO (adjust as needed) ---
REM Video Parameters
SET "IMAGE_H=180"
SET "IMAGE_W=320"
SET "NUM_CHANNELS=3"
SET "NUM_INPUT_FRAMES=14"  REM How many frames to condition on
SET "NUM_PREDICT_FRAMES=1" REM Script predicts 1 future frame's features
SET "FRAME_SKIP=1"

REM Feature Dimensions
SET "INITIAL_CNN_FEATURE_DIM=128"
SET "WUBU_S_OUTPUT_DIM=64"
SET "WUBU_T_OUTPUT_DIM=128"

REM GAAD Parameters
SET "GAAD_NUM_REGIONS=7"
SET "GAAD_ROI_H=5"
SET "GAAD_ROI_W=5"
SET "GAAD_FEAT_DIM=64"
SET "GAAD_DECOMP_TYPE=hybrid" REM spiral, subdivide, hybrid

REM Diffusion Parameters
SET "TIMESTEPS=50"
SET "DIFFUSION_TIME_EMBEDDING_DIM=64"
SET "BETA_SCHEDULE=cosine" REM linear or cosine
SET "COSINE_S=0.008"
SET "PHI_TIME_DIFFUSION_SCALE=1.0"
SET "PHI_TIME_BASE_FREQ=10000.0"

REM WuBu-S Parameters
SET "WUBU_S_NUM_LEVELS=3"
SET "WUBU_S_HYPERBOLIC_DIMS=64 48 32" REM Space-separated list, length must match WUBU_S_NUM_LEVELS
SET "WUBU_S_INITIAL_CURVATURES=1.0 0.7 0.5" REM Space-separated list, length must match WUBU_S_NUM_LEVELS
SET "WUBU_S_USE_ROTATION=false" REM Set to true to enable
SET "WUBU_S_PHI_CURVATURE=true" REM Set to true to enable
SET "WUBU_S_PHI_ROT_INIT=false" REM Set to true to enable

REM WuBu-T Parameters
SET "WUBU_T_NUM_LEVELS=3"
SET "WUBU_T_HYPERBOLIC_DIMS=64 94 128" REM Space-separated list, length must match WUBU_T_NUM_LEVELS
SET "WUBU_T_INITIAL_CURVATURES=1.0 0.7 0.5" REM Space-separated list, length must match WUBU_T_NUM_LEVELS
SET "WUBU_T_USE_ROTATION=false" REM Set to true to enable
SET "WUBU_T_PHI_CURVATURE=true" REM Set to true to enable
SET "WUBU_T_PHI_ROT_INIT=false" REM Set to true to enable

SET "WUBU_DROPOUT=0.15"

REM Training Parameters
SET "EPOCHS=75"
SET "GLOBAL_BATCH_SIZE=32"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
SET "LEARNING_RATE=0.07"
SET "RISGD_MAX_GRAD_NORM=9001.0"
SET "GLOBAL_MAX_GRAD_NORM=9001.0" REM 0 to disable global clip

REM --- Script Execution Arguments ---
SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --video_data_path "%VIDEO_DATA_PATH%""
REM SET "SCRIPT_ARGS=%SCRIPT_ARGS% --single_video_roll" REM Uncomment if VIDEO_DATA_PATH is a single file

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR%""
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --image_h %IMAGE_H%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --image_w %IMAGE_W%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_channels %NUM_CHANNELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_input_frames %NUM_INPUT_FRAMES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_predict_frames %NUM_PREDICT_FRAMES%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --frame_skip %FRAME_SKIP%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --initial_cnn_feature_dim %INITIAL_CNN_FEATURE_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_output_dim %WUBU_S_OUTPUT_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_output_dim %WUBU_T_OUTPUT_DIM%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_num_regions %GAAD_NUM_REGIONS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_region_roi_output_h %GAAD_ROI_H%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_region_roi_output_w %GAAD_ROI_W%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_region_feature_dim %GAAD_FEAT_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --gaad_decomposition_type %GAAD_DECOMP_TYPE%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --timesteps %TIMESTEPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --diffusion_time_embedding_dim %DIFFUSION_TIME_EMBEDDING_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --beta_schedule %BETA_SCHEDULE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --cosine_s %COSINE_S%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --phi_time_diffusion_scale %PHI_TIME_DIFFUSION_SCALE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --phi_time_base_freq %PHI_TIME_BASE_FREQ%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_num_levels %WUBU_S_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_hyperbolic_dims %WUBU_S_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_initial_curvatures %WUBU_S_INITIAL_CURVATURES%"
IF "%WUBU_S_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_use_rotation"
IF "%WUBU_S_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_phi_influence_curvature"
IF "%WUBU_S_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_s_phi_influence_rotation_init"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_num_levels %WUBU_T_NUM_LEVELS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_hyperbolic_dims %WUBU_T_HYPERBOLIC_DIMS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_initial_curvatures %WUBU_T_INITIAL_CURVATURES%"
IF "%WUBU_T_USE_ROTATION%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_use_rotation"
IF "%WUBU_T_PHI_CURVATURE%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_phi_influence_curvature"
IF "%WUBU_T_PHI_ROT_INIT%"=="true" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_t_phi_influence_rotation_init"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_dropout %WUBU_DROPOUT%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --epochs %EPOCHS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --batch_size %BATCH_SIZE_PER_GPU%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --grad_accum_steps %GRAD_ACCUM_STEPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --learning_rate %LEARNING_RATE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --risgd_max_grad_norm %RISGD_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --global_max_grad_norm %GLOBAL_MAX_GRAD_NORM%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --log_interval 2"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --save_interval 500000000"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --seed 42"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --use_amp"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --num_workers 0"
REM SET "SCRIPT_ARGS=%SCRIPT_ARGS% --detect_anomaly" REM Uncomment for debugging NaNs

REM --- Q-Controller & WandB ---
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_controller_enabled"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wandb --wandb_project WuBuGAADPhiDemo_v04"

REM --- Environment Activation & Execution ---
ECHO ======================================================
ECHO WuBu GAAD Phi Diffusion - FUNCTIONAL DEMO V0.04
ECHO Python: %PYTHON_EXE%
ECHO Video Data: %VIDEO_DATA_PATH%
ECHO Checkpoints: %CHECKPOINT_OUTPUT_DIR%
ECHO Batch Size per GPU: %BATCH_SIZE_PER_GPU%
ECHO NPROC_PER_NODE: %NPROC_PER_NODE%
ECHO ======================================================
ECHO.

IF NOT EXIST "%PYTHON_EXE%" (
    ECHO ERROR: Python executable not found at %PYTHON_EXE%
    ECHO Please ensure your Python environment is correctly set up and PYTHON_EXE variable is correct.
    GOTO :End
)

REM Attempt to activate venv if it exists where PYTHON_EXE points
SET "VENV_ACTIVATE_PATH="
FOR %%F IN ("%PYTHON_EXE%") DO SET "VENV_ACTIVATE_PATH=%%~dpFactivate.bat"

IF EXIST "%VENV_ACTIVATE_PATH%" (
    ECHO Activating virtual environment: %VENV_ACTIVATE_PATH%
    CALL "%VENV_ACTIVATE_PATH%"
    IF ERRORLEVEL 1 (
        ECHO WARNING: Failed to activate virtual environment, but will proceed.
    )
) ELSE (
    ECHO Virtual environment activate.bat not found at %VENV_ACTIVATE_PATH%. Assuming environment is already active or Python is in global PATH.
)
ECHO.

ECHO Creating checkpoint directory if it doesn't exist: %CHECKPOINT_OUTPUT_DIR%
IF NOT EXIST "%CHECKPOINT_OUTPUT_DIR%" (
    MKDIR "%CHECKPOINT_OUTPUT_DIR%"
)
ECHO.
ECHO Creating base data directory if it doesn't exist: %DATA_DIR_BASE%
IF NOT EXIST "%DATA_DIR_BASE%" (
    MKDIR "%DATA_DIR_BASE%"
)
ECHO.

ECHO Starting training script: %SCRIPT_NAME%
ECHO Script path: %FULL_SCRIPT_PATH%
ECHO Arguments: %SCRIPT_ARGS%
ECHO.

IF %NPROC_PER_NODE% EQU 1 (
    ECHO Using direct Python execution for single GPU/CPU...
    "%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" %SCRIPT_ARGS%
) ELSE (
    ECHO Using torchrun with NPROC_PER_NODE=%NPROC_PER_NODE%
    REM For multi-GPU, torchrun is preferred. Ensure PyTorch Distributed is installed.
    "%PYTHON_EXE%" -m torch.distributed.run --nproc_per_node=%NPROC_PER_NODE% --master_addr=%MASTER_ADDR% --master_port=%MASTER_PORT% --standalone --nnodes=1 "%FULL_SCRIPT_PATH%" %SCRIPT_ARGS%
)

SET "EXIT_CODE=%ERRORLEVEL%"
ECHO.
IF %EXIT_CODE% NEQ 0 (
    ECHO *******************************************
    ECHO *  DEMO SCRIPT FAILED with exit code %EXIT_CODE%  *
    ECHO *******************************************
) ELSE (
    ECHO *******************************************
    ECHO *  DEMO SCRIPT FINISHED successfully  *
    ECHO *******************************************
)

:End
ECHO Batch script execution finished.
IF DEFINED PROMPT_AFTER_RUN (
    PAUSE
) ELSE (
    TIMEOUT /T 10 /NOBREAK >nul
)
ENDLOCAL