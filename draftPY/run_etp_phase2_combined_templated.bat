@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM =====================================================================
REM Configuration
REM =====================================================================
SET "PROJECT_ROOT=%~dp0.."
SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT_NAME=etp_combined_phase2.py"
SET "FULL_SCRIPT_PATH=!SCRIPT_DIR!!SCRIPT_NAME!"
SET "TRAINING_MODE=ala"
REM SET "TRAINING_MODE=dissection_mimicry"
SET "PYTHON_EXE=python"
SET "EMBEDDINGS_FILE_A_ALA="
SET "EMBEDDINGS_FILE_B_ALA="
SET "DISSECTED_DATA_FILE_TRAIN_DM="
SET "DISSECTED_DATA_FILE_VAL_DM="
SET "LOAD_CHECKPOINT="
SET "CHECKPOINT_BASE_DIR=!PROJECT_ROOT!\etp_universal_checkpoints"
SET "CHECKPOINT_OUTPUT_DIR_CURRENT_MODE="

REM =====================================================================
REM Execution Flow
REM =====================================================================
CALL :FindPython
IF !ERRORLEVEL! NEQ 0 EXIT /B !ERRORLEVEL!
CALL :SetupCheckpointPaths
IF !ERRORLEVEL! NEQ 0 EXIT /B !ERRORLEVEL!
CALL :FindLastCheckpointToLoad
IF !ERRORLEVEL! EQU 0 (
    ECHO INFO: Found checkpoint to load: !LOAD_CHECKPOINT!
) ELSE (
    ECHO INFO: No suitable checkpoint found. Starting fresh or from Phase 1.
)
CALL :ValidateEnvironment
IF !ERRORLEVEL! NEQ 0 EXIT /B !ERRORLEVEL!
CALL :BuildArguments
IF !ERRORLEVEL! NEQ 0 EXIT /B !ERRORLEVEL!
CALL :RunTraining
ENDLOCAL
EXIT /B %ERRORLEVEL%

REM =====================================================================
REM Subroutines
REM =====================================================================

:FindPython
    IF EXIST "!PROJECT_ROOT!\venv\Scripts\python.exe" (
        SET "PYTHON_EXE=!PROJECT_ROOT!\venv\Scripts\python.exe"
        ECHO INFO: Using venv Python: !PYTHON_EXE!
        EXIT /B 0
    )
    IF EXIST "!PROJECT_ROOT!\.venv\Scripts\python.exe" (
        SET "PYTHON_EXE=!PROJECT_ROOT!\.venv\Scripts\python.exe"
        ECHO INFO: Using .venv Python: !PYTHON_EXE!
        EXIT /B 0
    )
    ECHO INFO: Using system Python: !PYTHON_EXE!
EXIT /B 0

:SetupCheckpointPaths
    SET "CHECKPOINT_OUTPUT_DIR_CURRENT_MODE=!CHECKPOINT_BASE_DIR!\!TRAINING_MODE!"
    IF NOT EXIST "!CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!" (
        MKDIR "!CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!"
        IF !ERRORLEVEL! NEQ 0 (ECHO ERROR: Failed to create checkpoint directory: !CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!; EXIT /B 1)
        ECHO INFO: Created checkpoint directory: !CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!
    )
EXIT /B 0

:FindLastCheckpointToLoad
    SET "BEST_CKPT_PATTERN=!CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!\ckpt_!TRAINING_MODE!_best_val*.pth.tar"
    FOR %%F IN ("!BEST_CKPT_PATTERN!") DO (IF EXIST "%%F" (SET "LOAD_CHECKPOINT=%%F"; ECHO INFO: Found best checkpoint for !TRAINING_MODE! mode.; EXIT /B 0))
    FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "!CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!\ckpt_!TRAINING_MODE!_ep*_gs*.pth.tar" 2^>nul') DO (SET "LOAD_CHECKPOINT=!CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!\%%F"; ECHO INFO: Found latest epoch checkpoint for !TRAINING_MODE! mode.; EXIT /B 0)
    SET "CHECKPOINT_LOAD_DIR_PHASE1=!PROJECT_ROOT!\etp_phase1_reconstruction\checkpoints_phase1_rec_templated"
    FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "!CHECKPOINT_LOAD_DIR_PHASE1!\checkpoint_p1_*.pth.tar" 2^>nul') DO (SET "LOAD_CHECKPOINT=!CHECKPOINT_LOAD_DIR_PHASE1!\%%F"; ECHO INFO: No !TRAINING_MODE! checkpoint. Found latest Phase 1 checkpoint.; EXIT /B 0)
    ECHO WARNING: No checkpoints found. Will start from scratch.
EXIT /B 1

:ValidateEnvironment
    IF NOT EXIST "!PYTHON_EXE!" (ECHO ERROR: Python not found: !PYTHON_EXE!; EXIT /B 1)
    IF NOT EXIST "!FULL_SCRIPT_PATH!" (ECHO ERROR: Script not found: !FULL_SCRIPT_PATH!; EXIT /B 1)
    IF "!TRAINING_MODE!"=="ala" (
        SET "EMBEDDINGS_FILE_A_ALA=!PROJECT_ROOT!\etp_corpus_A_deepseek_r1_embeddings.npz"
        SET "EMBEDDINGS_FILE_B_ALA=!PROJECT_ROOT!\etp_corpus_B_deepseek_r1_embeddings.npz"
        IF NOT EXIST "!EMBEDDINGS_FILE_A_ALA!" (ECHO ERROR: Missing ALA embeddings_file_A: !EMBEDDINGS_FILE_A_ALA!; EXIT /B 1)
        IF NOT EXIST "!EMBEDDINGS_FILE_B_ALA!" (ECHO ERROR: Missing ALA embeddings_file_B: !EMBEDDINGS_FILE_B_ALA!; EXIT /B 1)
    ) ELSE IF "!TRAINING_MODE!"=="dissection_mimicry" (
        SET "DISSECTED_DATA_FILE_TRAIN_DM=!PROJECT_ROOT!\dissected_data\deepseek_r1_dissected_real_A_train.npz"
        IF NOT EXIST "!DISSECTED_DATA_FILE_TRAIN_DM!" (ECHO ERROR: Missing DM dissected_data_file_train: !DISSECTED_DATA_FILE_TRAIN_DM!; EXIT /B 1)
    ) ELSE (ECHO ERROR: Invalid TRAINING_MODE: !TRAINING_MODE!; EXIT /B 1)
EXIT /B 0

:BuildArguments
    REM --- Clear and start building the argument string ---
    SET "SCRIPT_ARGS="
    
    REM --- Helper variables for JSON arguments to ensure correct quoting ---
    SET "ARG_WUBU_CORE_CONFIG_JSON=null"
    SET "ARG_TRANSFUSION_HEAD_CONFIG_JSON={}"
    SET "ARG_MAIN_DECODER_CONFIG_JSON={}"
    SET "ARG_OPTIM_WUBU_CORE_JSON={}"
    SET "ARG_OPTIM_MLPS_JSON={}"
    SET "ARG_Q_WUBU_CORE_JSON=null"
    SET "ARG_Q_MLPS_JSON=null"
    
    REM --- Build the command argument by argument ---
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --training_mode "!TRAINING_MODE!""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --checkpoint_dir "!CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!""
    IF DEFINED LOAD_CHECKPOINT SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_checkpoint "!LOAD_CHECKPOINT!""

    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --ds_r1_embedding_dim 1536"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_initial_tangent_dim 256"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_core_config_json "!ARG_WUBU_CORE_CONFIG_JSON!""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --transfusion_head_config_json "!ARG_TRANSFUSION_HEAD_CONFIG_JSON!""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --main_decoder_config_json "!ARG_MAIN_DECODER_CONFIG_JSON!""
    
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --epochs 10"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --batch_size 16"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lr_sphere_wubu_core 1e-4"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lr_sphere_mlps 1e-4"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_rec 1.0"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_vsp 0.01"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --log_interval 10"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --save_interval 100"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_project "ETP_Universal_Output""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wandb_run_name "!TRAINING_MODE!_run_!RANDOM!""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optimizer_kwargs_wubu_core_json "!ARG_OPTIM_WUBU_CORE_JSON!""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optimizer_kwargs_mlps_json "!ARG_OPTIM_MLPS_JSON!""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_controller_enabled true"
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_config_sphere_wubu_core_json "!ARG_Q_WUBU_CORE_JSON!""
    SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_config_sphere_mlps_json "!ARG_Q_MLPS_JSON!""
    
    IF "!TRAINING_MODE!"=="ala" (
        ECHO INFO: Building arguments for ALA mode.
        SET "ARG_OPTIM_DISC_JSON={}"
        SET "ARG_Q_DISC_JSON=null"
        SET "ARG_DISC_HIDDEN_DIMS_JSON=[256, 128]"

        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --embeddings_file_A "!EMBEDDINGS_FILE_A_ALA!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --embeddings_file_B "!EMBEDDINGS_FILE_B_ALA!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lr_discriminator 2e-4"
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --optimizer_kwargs_discriminator_json "!ARG_OPTIM_DISC_JSON!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --q_config_discriminator_json "!ARG_Q_DISC_JSON!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --disc_hidden_dims_json "!ARG_DISC_HIDDEN_DIMS_JSON!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --lambda_ala 0.1"
    ) ELSE IF "!TRAINING_MODE!"=="dissection_mimicry" (
        ECHO INFO: Building arguments for Dissection Mimicry mode.
        SET "ARG_DM_TARGET_TEACHER_JSON=[5, 15, 25]"
        SET "ARG_DM_STUDENT_WUBU_JSON=[0, 1, 2]"
        SET "ARG_DM_AUX_DECODER_JSON={\"num_mlp_layers\":1, \"mlp_hidden_dim_ratio\":1.0}"
        SET "ARG_DM_LAMBDA_INTERMEDIATE_JSON={\"teacher_layer_5_pooled\":0.1, \"teacher_layer_15_pooled\":0.1, \"teacher_layer_25_pooled\":0.1}"

        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dissected_data_file_train "!DISSECTED_DATA_FILE_TRAIN_DM!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dm_target_teacher_layers_json "!ARG_DM_TARGET_TEACHER_JSON!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dm_student_wubu_levels_json "!ARG_DM_STUDENT_WUBU_JSON!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dm_aux_decoder_config_json "!ARG_DM_AUX_DECODER_JSON!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dm_lambda_distill_intermediate_json "!ARG_DM_LAMBDA_INTERMEDIATE_JSON!""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dm_lambda_distill_final 1.0"
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dm_wubu_core_processes_sequences false"
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dm_pool_teacher_states true"
    )
EXIT /B 0

:RunTraining
    ECHO.
    ECHO ======================================================
    ECHO          STARTING ETP UNIVERSAL TRAINING
    ECHO ======================================================
    ECHO Mode: !TRAINING_MODE!
    ECHO Python: !PYTHON_EXE!
    ECHO Script: !FULL_SCRIPT_PATH!
    ECHO Arguments:
    ECHO !SCRIPT_ARGS!
    ECHO Checkpoint Output Dir: !CHECKPOINT_OUTPUT_DIR_CURRENT_MODE!
    IF DEFINED LOAD_CHECKPOINT ECHO Loading Checkpoint: !LOAD_CHECKPOINT!
    ECHO.

    "!PYTHON_EXE!" "!FULL_SCRIPT_PATH!" !SCRIPT_ARGS!
    SET "TRAINING_EXIT_CODE=!ERRORLEVEL!"

    ECHO.
    ECHO ======================================================
    IF !TRAINING_EXIT_CODE! EQU 0 (
        ECHO          TRAINING COMPLETED SUCCESSFULLY
    ) ELSE (
        ECHO          TRAINING FAILED (Exit Code: !TRAINING_EXIT_CODE!)
    )
    ECHO ======================================================
    ECHO.
    PAUSE
EXIT /B !TRAINING_EXIT_CODE!