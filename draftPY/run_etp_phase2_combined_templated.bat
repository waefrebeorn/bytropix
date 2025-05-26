@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM =====================================================================
REM Project and Python Setup
REM =====================================================================
SET "PROJECT_ROOT=%~dp0"
IF EXIST "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
) ELSE IF EXIST "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    SET "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) ELSE (
    SET "PYTHON_EXE=python"
)
SET "SCRIPT_DIR=%PROJECT_ROOT%"
SET "SCRIPT_NAME=etp_combined_phase2.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

REM =====================================================================
REM DDP Configuration
REM =====================================================================
SET "NPROC_PER_NODE=1"
SET "MASTER_ADDR=localhost"
SET "MASTER_PORT=29502"

REM =====================================================================
REM Path Configuration
REM =====================================================================
SET "CHECKPOINT_OUTPUT_DIR_PHASE2=%PROJECT_ROOT%\etp_phase2_ala\checkpoints_phase2_ala_templated"
SET "CHECKPOINT_LOAD_DIR_PHASE1=%PROJECT_ROOT%\etp_phase1_reconstruction\checkpoints_phase1_rec_templated"
SET "EMBEDDINGS_FILE_A=%PROJECT_ROOT%\dummy_corpus_A_embeddings.npz"
SET "EMBEDDINGS_FILE_B=%PROJECT_ROOT%\dummy_corpus_B_embeddings.npz"

SET "LOAD_CHECKPOINT_PHASE1="
SET "LATEST_EPOCH_CKPT_PHASE1="
REM Find the latest Phase 1 checkpoint to load
FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "%CHECKPOINT_LOAD_DIR_PHASE1%\checkpoint_p1_epoch*_step*.pth.tar"') DO (
    SET "LATEST_EPOCH_CKPT_PHASE1=%%F"
    GOTO FoundLatestPhase1ForPhase2
)
:FoundLatestPhase1ForPhase2
IF DEFINED LATEST_EPOCH_CKPT_PHASE1 (
    SET "LOAD_CHECKPOINT_PHASE1=%CHECKPOINT_LOAD_DIR_PHASE1%\!LATEST_EPOCH_CKPT_PHASE1!"
) ELSE (
    ECHO WARNING: No Phase 1 checkpoint found in %CHECKPOINT_LOAD_DIR_PHASE1%. Phase 2 will start from scratch or fail if load is required.
)

SET "LOAD_CHECKPOINT_PHASE2="
SET "LATEST_EPOCH_CKPT_PHASE2="
REM Optionally, load a Phase 2 checkpoint if resuming a Phase 2 run
IF EXIST "%CHECKPOINT_OUTPUT_DIR_PHASE2%\ckpt_p2_best_ep*_gs*.pth.tar" (
     SET "LOAD_CHECKPOINT_PHASE2=%CHECKPOINT_OUTPUT_DIR_PHASE2%\ckpt_p2_best.pth.tar"
) ELSE (
    FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "%CHECKPOINT_OUTPUT_DIR_PHASE2%\ckpt_p2_ep*_gs*.pth.tar"') DO (
        SET "LATEST_EPOCH_CKPT_PHASE2=%%F"
        GOTO FoundLatestEpochCkpt_Phase2
    )
    :FoundLatestEpochCkpt_Phase2
    IF DEFINED LATEST_EPOCH_CKPT_PHASE2 (
        SET "LOAD_CHECKPOINT_PHASE2=%CHECKPOINT_OUTPUT_DIR_PHASE2%\!LATEST_EPOCH_CKPT_PHASE2!"
    )
)
REM Prioritize resuming Phase 2 checkpoint if available, otherwise load Phase 1
IF DEFINED LOAD_CHECKPOINT_PHASE2 (
    SET "LOAD_CHECKPOINT=!LOAD_CHECKPOINT_PHASE2!"
    ECHO Resuming Phase 2 from: !LOAD_CHECKPOINT!
) ELSE (
    SET "LOAD_CHECKPOINT=!LOAD_CHECKPOINT_PHASE1!"
    ECHO Starting Phase 2, loading from Phase 1: !LOAD_CHECKPOINT!
)

REM =====================================================================
REM ETP Sphere Model Configuration (Same as Phase 1)
REM =====================================================================
SET "DS_R1_EMBEDDING_DIM=768"
SET "WUBU_INITIAL_TANGENT_DIM=256"
SET "HEAD_MLP_LAYERS=2"
SET "DECODER_MLP_LAYERS=2"
SET "WUBU_CORE_CONFIG_JSON="

REM =====================================================================
REM Discriminator Configuration
REM =====================================================================
SET "DISC_HIDDEN_DIMS_JSON=[256, 128]"
SET "DISC_ACTIVATION_FN=leaky_relu"
SET "DISC_USE_SPECTRAL_NORM=true"

REM =====================================================================
REM Loss Weights Configuration (Phase 2)
REM =====================================================================
SET "LAMBDA_ALA=0.1"
SET "LAMBDA_REC=1.0"
SET "LAMBDA_VSP=0.01"

REM =====================================================================
REM Training Hyperparameters (Same as Phase 1, but epochs might differ)
REM =====================================================================
SET "EPOCHS=10"
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
SET "LR_SPHERE_WUBU_CORE=5e-5"
SET "LR_SPHERE_MLPS=5e-5"
SET "LR_DISCRIMINATOR=1e-4"
SET "OPTIMIZER_KWARGS_WUBU_CORE_JSON={}"
SET "OPTIMIZER_KWARGS_MLPS_JSON={}"
SET "OPTIMIZER_KWARGS_DISCRIMINATOR_JSON={}"

REM =====================================================================
REM Q-Controller Configuration
REM =====================================================================
SET "Q_CONTROLLER_ENABLED=false"
SET "Q_CONFIG_SPHERE_WUBU_CORE_JSON="
SET "Q_CONFIG_SPHERE_MLPS_JSON="
SET "Q_CONFIG_DISCRIMINATOR_JSON="

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
SET "WANDB_PROJECT=ETP_Phase2_ALA_Templated"
SET "WANDB_RUN_NAME=phase2_ala_templated_run"

REM =====================================================================
REM Script Argument Assembly
REM =====================================================================
SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --embeddings_file_A "%EMBEDDINGS_FILE_A%""
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --embeddings_file_B "%EMBEDDINGS_FILE_B%""
IF DEFINED LOAD_CHECKPOINT (
    IF NOT "%LOAD_CHECKPOINT%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --load_checkpoint "%LOAD_CHECKPOINT%""
)
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --checkpoint_dir "%CHECKPOINT_OUTPUT_DIR_PHASE2%""

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --ds_r1_embedding_dim %DS_R1_EMBEDDING_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_initial_tangent_dim %WUBU_INITIAL_TANGENT_DIM%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --head_mlp_layers %HEAD_MLP_LAYERS%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --decoder_mlp_layers %DECODER_MLP_LAYERS%"
IF DEFINED WUBU_CORE_CONFIG_JSON (
    IF NOT "%WUBU_CORE_CONFIG_JSON%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --wubu_core_config_json "%WUBU_CORE_CONFIG_JSON%""
)

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_hidden_dims_json "%DISC_HIDDEN_DIMS_JSON%""
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --disc_activation_fn %DISC_ACTIVATION_FN%"
IF /I "%DISC_USE_SPECTRAL_NORM%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_use_spectral_norm"

SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lambda_ala %LAMBDA_ALA%"
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
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --lr_discriminator %LR_DISCRIMINATOR%"
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --optimizer_kwargs_wubu_core_json "%OPTIMIZER_KWARGS_WUBU_CORE_JSON%""
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --optimizer_kwargs_mlps_json "%OPTIMIZER_KWARGS_MLPS_JSON%""
SET "SCRIPT_ARGS=%SCRIPT_ARGS% --optimizer_kwargs_discriminator_json "%OPTIMIZER_KWARGS_DISCRIMINATOR_JSON%""

IF /I "%Q_CONTROLLER_ENABLED%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled"
IF DEFINED Q_CONFIG_SPHERE_WUBU_CORE_JSON (
    IF NOT "%Q_CONFIG_SPHERE_WUBU_CORE_JSON%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_config_sphere_wubu_core_json "%Q_CONFIG_SPHERE_WUBU_CORE_JSON%""
)
IF DEFINED Q_CONFIG_SPHERE_MLPS_JSON (
    IF NOT "%Q_CONFIG_SPHERE_MLPS_JSON%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_config_sphere_mlps_json "%Q_CONFIG_SPHERE_MLPS_JSON%""
)
IF DEFINED Q_CONFIG_DISCRIMINATOR_JSON (
    IF NOT "%Q_CONFIG_DISCRIMINATOR_JSON%"=="" SET "SCRIPT_ARGS=%SCRIPT_ARGS% --q_config_discriminator_json "%Q_CONFIG_DISCRIMINATOR_JSON%""
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
ECHO ETP Phase 2 (Adversarial Latent Alignment) - Combined Script
ECHO Python: %PYTHON_EXE%
ECHO Script Name: %SCRIPT_NAME%
ECHO Checkpoints (Save to): %CHECKPOINT_OUTPUT_DIR_PHASE2%
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
IF NOT EXIST "%CHECKPOINT_OUTPUT_DIR_PHASE2%" MKDIR "%CHECKPOINT_OUTPUT_DIR_PHASE2%"
IF NOT EXIST "%CHECKPOINT_LOAD_DIR_PHASE1%" (
    ECHO WARNING: Phase 1 checkpoint directory %CHECKPOINT_LOAD_DIR_PHASE1% does not exist. This might be an issue if loading from Phase 1.
)
ECHO.

ECHO Starting training script: %SCRIPT_NAME%
ECHO.

REM For Phase 2, DDP is less likely for small models, but template kept.
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