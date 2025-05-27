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
SET "SCRIPT_NAME=etp_combined_phase1.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

REM =====================================================================
REM DDP Configuration
REM =====================================================================
SET "NPROC_PER_NODE=1"
SET "MASTER_ADDR=localhost"
SET "MASTER_PORT=29501"

REM =====================================================================
REM Path Configuration
REM =====================================================================
SET "CHECKPOINT_OUTPUT_DIR=%PROJECT_ROOT%\etp_phase1_reconstruction\checkpoints_phase1_rec_templated"
SET "EMBEDDINGS_FILE_A=%PROJECT_ROOT%\etp_corpus_A_deepseek_r1_embeddings.npz"
SET "EMBEDDINGS_FILE_B=%PROJECT_ROOT%\etp_corpus_B_deepseek_r1_embeddings.npz"
SET "LOAD_CHECKPOINT="
SET "FOUND_CKPT_NAME="

REM Try to find a specific "best" checkpoint first using a pattern
ECHO INFO: Looking for "best" Phase 1 checkpoint...
FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "%CHECKPOINT_OUTPUT_DIR%\checkpoint_p1_best.pth.tar" 2^>nul') DO (
    SET "FOUND_CKPT_NAME=%%F"
    GOTO GotPhase1BestCkpt
)
:GotPhase1BestCkpt
IF DEFINED FOUND_CKPT_NAME (
    IF NOT "!FOUND_CKPT_NAME!"=="" (
        SET "LOAD_CHECKPOINT=%CHECKPOINT_OUTPUT_DIR%\!FOUND_CKPT_NAME!"
        ECHO INFO: Found "best" Phase 1 checkpoint: !LOAD_CHECKPOINT!
    )
)

REM If "best" not found, look for the latest epoch checkpoint
IF NOT DEFINED LOAD_CHECKPOINT (
    ECHO INFO: "Best" Phase 1 checkpoint not found. Looking for latest epoch checkpoint...
    SET "FOUND_CKPT_NAME=" REM Reset for this search
    FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "%CHECKPOINT_OUTPUT_DIR%\checkpoint_p1_epoch*_step*.pth.tar" 2^>nul') DO (
        SET "FOUND_CKPT_NAME=%%F"
        GOTO GotPhase1LatestEpochCkpt
    )
    :GotPhase1LatestEpochCkpt
    IF DEFINED FOUND_CKPT_NAME (
        IF NOT "!FOUND_CKPT_NAME!"=="" (
            SET "LOAD_CHECKPOINT=%CHECKPOINT_OUTPUT_DIR%\!FOUND_CKPT_NAME!"
            ECHO INFO: Found latest epoch Phase 1 checkpoint: !LOAD_CHECKPOINT!
        )
    )
)

IF NOT DEFINED LOAD_CHECKPOINT (
    ECHO WARNING: No Phase 1 checkpoint found to load. Starting from scratch or Python script might fail.
)

REM Ensure dummy embedding files are generated (rest of this section is fine)
IF NOT EXIST "%EMBEDDINGS_FILE_A%" (
    ECHO WARNING: "%EMBEDDINGS_FILE_A%" not found. Attempting to generate dummy embeddings...
    IF EXIST "%PROJECT_ROOT%\etp_common\etp_embedding_extractor.py" (
        "%PYTHON_EXE%" "%PROJECT_ROOT%\etp_common\etp_embedding_extractor.py"
        IF ERRORLEVEL 1 (
            ECHO ERROR: Failed to generate dummy embeddings for A. Please check etp_embedding_extractor.py
            GOTO :End
        )
    ) ELSE (
        ECHO ERROR: Dummy embedding generator script not found at %PROJECT_ROOT%\etp_common\etp_embedding_extractor.py
        GOTO :End
    )
)
IF NOT EXIST "%EMBEDDINGS_FILE_B%" (
     ECHO WARNING: "%EMBEDDINGS_FILE_B%" not found. This might be an issue if corpus B is expected.
)

REM ... (Rest of Phase 1 script: Model Config, Loss, Training, Optim, Q-Ctrl, Logging, WandB, Arg Assembly, Echo, Execution)
REM (All SET "SCRIPT_ARGS=!SCRIPT_ARGS! ..." lines should be fine)
REM (The MKDIR check for CHECKPOINT_OUTPUT_DIR is fine)
REM (The final execution block is fine)

SET "DS_R1_EMBEDDING_DIM=1536"
SET "WUBU_INITIAL_TANGENT_DIM=256"
SET "HEAD_MLP_LAYERS=2"
SET "DECODER_MLP_LAYERS=2"
SET "WUBU_CORE_CONFIG_JSON="
SET "LAMBDA_REC=1.0"
SET "LAMBDA_VSP=0.0"
SET "EPOCHS=5000"
SET "GLOBAL_BATCH_SIZE=1024"
SET "BATCH_SIZE_PER_GPU=%GLOBAL_BATCH_SIZE%"
IF %NPROC_PER_NODE% GTR 1 (
    SET /A BATCH_SIZE_PER_GPU = GLOBAL_BATCH_SIZE / NPROC_PER_NODE
    IF !BATCH_SIZE_PER_GPU! LSS 1 SET "BATCH_SIZE_PER_GPU=1"
)
SET "GRAD_ACCUM_STEPS=1"
SET "GLOBAL_MAX_GRAD_NORM=1.0"
SET "USE_AMP=false"
SET "DEVICE=cuda"
SET "LR_SPHERE_WUBU_CORE=1e-4"
SET "LR_SPHERE_MLPS=1e-4"
SET "OPTIMIZER_KWARGS_WUBU_CORE_JSON={}"
SET "OPTIMIZER_KWARGS_MLPS_JSON={}"
SET "Q_CONTROLLER_ENABLED=false"
SET "Q_CONFIG_SPHERE_WUBU_CORE_JSON="
SET "Q_CONFIG_SPHERE_MLPS_JSON="
SET "LOG_INTERVAL=5"
SET "SAVE_INTERVAL=0"
SET "VAL_INTERVAL_EPOCHS=1"
SET "WANDB_ENABLED=false"
SET "WANDB_PROJECT=ETP_Phase1_Reconstruction_Templated"
SET "WANDB_RUN_NAME=phase1_rec_templated_run"

SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --embeddings_file_A "%EMBEDDINGS_FILE_A%""
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --embeddings_file_B "%EMBEDDINGS_FILE_B%""
IF DEFINED LOAD_CHECKPOINT (
    IF NOT "%LOAD_CHECKPOINT%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_checkpoint "%LOAD_CHECKPOINT%""
)
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR%""
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --ds_r1_embedding_dim %DS_R1_EMBEDDING_DIM%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_initial_tangent_dim %WUBU_INITIAL_TANGENT_DIM%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --head_mlp_layers %HEAD_MLP_LAYERS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --decoder_mlp_layers %DECODER_MLP_LAYERS%"
IF DEFINED WUBU_CORE_CONFIG_JSON (
    IF NOT "%WUBU_CORE_CONFIG_JSON%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_core_config_json "%WUBU_CORE_CONFIG_JSON%""
)
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_rec %LAMBDA_REC%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_vsp %LAMBDA_VSP%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --device %DEVICE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --epochs %EPOCHS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --batch_size %BATCH_SIZE_PER_GPU%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --grad_accum_steps %GRAD_ACCUM_STEPS%"
IF /I "%USE_AMP%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_amp"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --global_max_grad_norm %GLOBAL_MAX_GRAD_NORM%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lr_sphere_wubu_core %LR_SPHERE_WUBU_CORE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lr_sphere_mlps %LR_SPHERE_MLPS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optimizer_kwargs_wubu_core_json "%OPTIMIZER_KWARGS_WUBU_CORE_JSON%""
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optimizer_kwargs_mlps_json "%OPTIMIZER_KWARGS_MLPS_JSON%""
IF /I "%Q_CONTROLLER_ENABLED%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled"
IF DEFINED Q_CONFIG_SPHERE_WUBU_CORE_JSON (
    IF NOT "%Q_CONFIG_SPHERE_WUBU_CORE_JSON%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_config_sphere_wubu_core_json "%Q_CONFIG_SPHERE_WUBU_CORE_JSON%""
)
IF DEFINED Q_CONFIG_SPHERE_MLPS_JSON (
    IF NOT "%Q_CONFIG_SPHERE_MLPS_JSON%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_config_sphere_mlps_json "%Q_CONFIG_SPHERE_MLPS_JSON%""
)
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --log_interval %LOG_INTERVAL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --save_interval %SAVE_INTERVAL%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --val_interval_epochs %VAL_INTERVAL_EPOCHS%"
IF /I "%WANDB_ENABLED%"=="true" (
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_project %WANDB_PROJECT%"
    IF DEFINED WANDB_RUN_NAME (
        IF NOT "%WANDB_RUN_NAME%"=="" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_run_name "%WANDB_RUN_NAME%""
    )
)

ECHO ======================================================
ECHO ETP Phase 1 (Reconstruction) - Combined Script
ECHO Python: %PYTHON_EXE%
ECHO Script Name: %SCRIPT_NAME%
ECHO Script Path: %FULL_SCRIPT_PATH%
ECHO Checkpoints: %CHECKPOINT_OUTPUT_DIR%
ECHO Embeddings A: %EMBEDDINGS_FILE_A%
ECHO Embeddings B: %EMBEDDINGS_FILE_B%
ECHO Load Checkpoint: %LOAD_CHECKPOINT%
ECHO Device: %DEVICE%
ECHO AMP: %USE_AMP%
ECHO Epochs: %EPOCHS%
ECHO Batch Size (Global/PerGPU): %GLOBAL_BATCH_SIZE% / %BATCH_SIZE_PER_GPU%
ECHO Q-Controller: %Q_CONTROLLER_ENABLED%
ECHO WANDB: %WANDB_ENABLED% (Project: %WANDB_PROJECT%)
ECHO ======================================================
ECHO.

IF NOT EXIST "%PYTHON_EXE%" (
    ECHO ERROR: Python executable not found at %PYTHON_EXE%
    GOTO :End
)
IF NOT EXIST "%CHECKPOINT_OUTPUT_DIR%" (
    ECHO INFO: Creating checkpoint directory: %CHECKPOINT_OUTPUT_DIR%
    MKDIR "%CHECKPOINT_OUTPUT_DIR%"
    IF ERRORLEVEL 1 (
        ECHO ERROR: Failed to create checkpoint directory "%CHECKPOINT_OUTPUT_DIR%". Check permissions.
        GOTO :End
    )
)
ECHO.
ECHO Starting training script: %SCRIPT_NAME%
ECHO Full command: "%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
ECHO.
"%PYTHON_EXE%" "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
SET "EXIT_CODE=%ERRORLEVEL%"
ECHO.
IF %EXIT_CODE% NEQ 0 (
    ECHO * SCRIPT FAILED with exit code %EXIT_CODE% *
    IF %EXIT_CODE% EQU 2 ECHO   (Note: Python often returns exit code 2 if it cannot find the script file itself)
) ELSE (
    ECHO * SCRIPT FINISHED successfully *
)

:End
IF DEFINED PROMPT_AFTER_RUN ( PAUSE ) ELSE ( TIMEOUT /T 10 /NOBREAK >nul )
ENDLOCAL