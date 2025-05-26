@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM =====================================================================
REM Project and Python Setup
REM =====================================================================
REM Assuming this .bat file is in the root of the draftPY project,
REM and etp_combined_phase1.py is also in the root, or PROJECT_ROOT is adjusted.
SET "PROJECT_ROOT=%~dp0"
IF EXIST "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
) ELSE IF EXIST "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) ELSE (
    SET "PYTHON_EXE=python"
)
SET "SCRIPT_DIR=%PROJECT_ROOT%"
SET "SCRIPT_NAME=etp_combined_phase1.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

REM =====================================================================
REM DDP Configuration (Not typically used for these ETP phases, but kept for template consistency)
REM =====================================================================
SET "NPROC_PER_NODE=1"
SET "MASTER_ADDR=localhost"
SET "MASTER_PORT=29501"

REM =====================================================================
REM Path Configuration
REM =====================================================================
SET "CHECKPOINT_OUTPUT_DIR=%PROJECT_ROOT%\etp_phase1_reconstruction\checkpoints_phase1_rec_templated"
SET "EMBEDDINGS_FILE_A=%PROJECT_ROOT%\dummy_corpus_A_embeddings.npz"
SET "EMBEDDINGS_FILE_B=%PROJECT_ROOT%\dummy_corpus_B_embeddings.npz"
SET "LOAD_CHECKPOINT="
SET "BEST_CKPT_NAME=checkpoint_p1_best_epoch*_step*.pth.tar"
SET "LATEST_EPOCH_CKPT_NAME="

IF EXIST "%CHECKPOINT_OUTPUT_DIR%\%BEST_CKPT_NAME%" (
    REM This exact best name matching might be tricky with wildcards.
    REM For simplicity, let's assume a fixed "best" name or rely on latest epoch.
    SET "LOAD_CHECKPOINT=%CHECKPOINT_OUTPUT_DIR%\checkpoint_p1_best.pth.tar"
) ELSE (
    FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "%CHECKPOINT_OUTPUT_DIR%\checkpoint_p1_epoch*_step*.pth.tar"') DO (
        SET "LATEST_EPOCH_CKPT_NAME=%%F"
        GOTO FoundLatestEpochCkpt_Phase1
    )
    :FoundLatestEpochCkpt_Phase1
    IF DEFINED LATEST_EPOCH_CKPT_NAME (
        SET "LOAD_CHECKPOINT=%CHECKPOINT_OUTPUT_DIR%\!LATEST_EPOCH_CKPT_NAME!"
    )
)

REM Ensure dummy embedding files are generated if they don't exist
IF NOT EXIST "%EMBEDDINGS_FILE_A%" (
    ECHO "%EMBEDDINGS_FILE_A%" not found. Generating dummy embeddings...
    "%PYTHON_EXE%" "%PROJECT_ROOT%\etp_common\etp_embedding_extractor.py"
)
IF NOT EXIST "%EMBEDDINGS_FILE_B%" (
     ECHO "%EMBEDDINGS_FILE_B%" not found. Might need to generate if previous step failed for B.
)


REM =====================================================================
REM ETP Sphere Model Configuration
REM =====================================================================
SET "DS_R1_EMBEDDING_DIM=768"
SET "WUBU_INITIAL_TANGENT_DIM=256"
SET "HEAD_MLP_LAYERS=2"
SET "DECODER_MLP_LAYERS=2"
SET "WUBU_CORE_CONFIG_JSON="

REM =====================================================================
REM Loss Weights Configuration (Phase 1)
REM =====================================================================
SET "LAMBDA_REC=1.0"
SET "LAMBDA_VSP=0.0"

REM =====================================================================
REM Training Hyperparameters
REM =====================================================================
SET "EPOCHS=5"
SET "GLOBAL_BATCH_SIZE=4"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
SET "GLOBAL_MAX_GRAD_NORM=1.0"
SET "USE_AMP=false"
SET "DEVICE=cpu"

REM =====================================================================
REM Optimizer Settings
REM =====================================================================
SET "LR_SPHERE_WUBU_CORE=1e-4"
SET "LR_SPHERE_MLPS=1e-4"
SET "OPTIMIZER_KWARGS_WUBU_CORE_JSON={}"
SET "OPTIMIZER_KWARGS_MLPS_JSON={}"

REM =====================================================================
REM Q-Controller Configuration
REM =====================================================================
SET "Q_CONTROLLER_ENABLED=false"
SET "Q_CONFIG_SPHERE_WUBU_CORE_JSON="
SET "Q_CONFIG_SPHERE_MLPS_JSON="

REM =====================================================================
REM Logging, Sampling, Validation & Checkpointing
REM =====================================================================
SET "LOG_INTERVAL=5"
SET "SAVE_INTERVAL=0"
SET "VAL_INTERVAL_EPOCHS=1"

REM =====================================================================
REM Experiment Control (WandB)
REM =====================================================================
SET "WANDB_ENABLED=false"
SET "WANDB_PROJECT=ETP_Phase1_Reconstruction_Templated"
SET "WANDB_RUN_NAME=phase1_rec_templated_run"

REM =====================================================================
REM Script Argument Assembly
REM =====================================================================
SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --embeddings_file_A "%EMBEDDINGS_FILE_A%""
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --embeddings_file_B "%EMBEDDINGS_FILE_B%""
IF DEFINED LOAD_CHECKPOINT (
    IF NOT "%LOAD_CHECKPOINT%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --load_checkpoint "%LOAD_CHECKPOINT%""
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR%""

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --ds_r1_embedding_dim %DS_R1_EMBEDDING_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_initial_tangent_dim %WUBU_INITIAL_TANGENT_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --head_mlp_layers %HEAD_MLP_LAYERS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --decoder_mlp_layers %DECODER_MLP_LAYERS%"
IF DEFINED WUBU_CORE_CONFIG_JSON (
    IF NOT "%WUBU_CORE_CONFIG_JSON%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_core_config_json "%WUBU_CORE_CONFIG_JSON%""
)

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_rec %LAMBDA_REC%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_vsp %LAMBDA_VSP%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --device %DEVICE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --epochs %EPOCHS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --batch_size %BATCH_SIZE_PER_GPU%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --grad_accum_steps %GRAD_ACCUM_STEPS%"
IF /I "%USE_AMP%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_amp"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --global_max_grad_norm %GLOBAL_MAX_GRAD_NORM%"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lr_sphere_wubu_core %LR_SPHERE_WUBU_CORE%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lr_sphere_mlps %LR_SPHERE_MLPS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --optimizer_kwargs_wubu_core_json "%OPTIMIZER_KWARGS_WUBU_CORE_JSON%""
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --optimizer_kwargs_mlps_json "%OPTIMIZER_KWARGS_MLPS_JSON%""

IF /I "%Q_CONTROLLER_ENABLED%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled"
IF DEFINED Q_CONFIG_SPHERE_WUBU_CORE_JSON (
    IF NOT "%Q_CONFIG_SPHERE_WUBU_CORE_JSON%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_config_sphere_wubu_core_json "%Q_CONFIG_SPHERE_WUBU_CORE_JSON%""
)
IF DEFINED Q_CONFIG_SPHERE_MLPS_JSON (
    IF NOT "%Q_CONFIG_SPHERE_MLPS_JSON%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_config_sphere_mlps_json "%Q_CONFIG_SPHERE_MLPS_JSON%""
)

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --log_interval %LOG_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --save_interval %SAVE_INTERVAL%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --val_interval_epochs %VAL_INTERVAL_EPOCHS%"

IF /I "%WANDB_ENABLED%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_project %WANDB_PROJECT%"
    IF DEFINED WANDB_RUN_NAME (
        IF NOT "%WANDB_RUN_NAME%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_run_name "%WANDB_RUN_NAME%""
    )
)

REM =====================================================================
REM Pre-Run Echo
REM =====================================================================
ECHO ======================================================
ECHO ETP Phase 1 (Reconstruction) - Combined Script
ECHO Python: %PYTHON_EXE%
ECHO Script Name: %SCRIPT_NAME%
ECHO Checkpoints: %CHECKPOINT_OUTPUT_DIR%
ECHO Load Checkpoint: %LOAD_CHECKPOINT%
ECHO Device: %DEVICE%
ECHO AMP: %USE_AMP%
ECHO Epochs: %EPOCHS%
ECHO Batch Size (Global/PerGPU): %GLOBAL_BATCH_SIZE% / %BATCH_SIZE_PER_GPU%
ECHO Q-Controller: %Q_CONTROLLER_ENABLED%
ECHO WANDB: %WANDB_ENABLED% (Project: %WANDB_PROJECT%)
ECHO ======================================================
ECHO.

REM =====================================================================
REM Environment Setup and Execution
REM =====================================================================
IF NOT EXIST "%PYTHON_EXE%" (
    ECHO ERROR: Python executable not found at %PYTHON_EXE%
    GOTO :End
)
IF NOT EXIST "%FULL_SCRIPT_PATH%" (
    ECHO ERROR: Script not found at %FULL_SCRIPT_PATH%
    GOTO :End
)
IF NOT EXIST "%CHECKPOINT_OUTPUT_DIR%" MKDIR "%CHECKPOINT_OUTPUT_DIR%"
ECHO.

ECHO Starting training script: %SCRIPT_NAME%
ECHO.

REM For Phase 1, DDP is less likely to be used, so defaulting to single process.
REM You can adapt the DDP launch from your template if needed.
"%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!

SET "EXIT_CODE=%ERRORLEVEL%"
ECHO.
IF %EXIT_CODE% NEQ 0 (
    ECHO * SCRIPT FAILED with exit code %EXIT_CODE% *
) ELSE (
    ECHO * SCRIPT FINISHED successfully *
)

:End
IF DEFINED PROMPT_AFTER_RUN ( PAUSE ) ELSE ( TIMEOUT /T 5 /NOBREAK >nul )
ENDLOCAL