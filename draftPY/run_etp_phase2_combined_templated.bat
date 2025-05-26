@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM =====================================================================
REM Project and Python Setup
REM =====================================================================
SET "PROJECT_ROOT=%~dp0.."
SET "SCRIPT_DIR=%~dp0"
SET "SCRIPT_NAME=etp_combined_phase2.py"
SET "FULL_SCRIPT_PATH=!SCRIPT_DIR!!SCRIPT_NAME!"

REM --- Python executable discovery ---
SET "PYTHON_EXE=python"
CALL :FindPython
IF ERRORLEVEL 1 EXIT /B 1

REM =====================================================================
REM Checkpoint Loading Logic (Linear flow with minimal nesting)
REM =====================================================================
SET "LOAD_CHECKPOINT="
CALL :FindCheckpoints
IF ERRORLEVEL 1 (
    ECHO WARNING: No checkpoints found - starting from scratch
) ELSE (
    ECHO INFO: Using checkpoint: !LOAD_CHECKPOINT!
)

REM =====================================================================
REM Main Execution
REM =====================================================================
CALL :ValidateEnvironment
CALL :BuildArguments
CALL :RunTraining

ENDLOCAL
EXIT /B %ERRORLEVEL%

REM =====================================================================
REM Subroutines
REM =====================================================================

:FindPython
IF EXIST "!PROJECT_ROOT!\venv\Scripts\python.exe" (
    SET "PYTHON_EXE=!PROJECT_ROOT!\venv\Scripts\python.exe"
    ECHO INFO: Using venv Python
    EXIT /B 0
)
IF EXIST "!PROJECT_ROOT!\.venv\Scripts\python.exe" (
    SET "PYTHON_EXE=!PROJECT_ROOT!\.venv\Scripts\python.exe"
    ECHO INFO: Using .venv Python
    EXIT /B 0
)
ECHO INFO: Using system Python
EXIT /B 0

:FindCheckpoints
REM --- Phase 2 Best Checkpoint ---
SET "CHECKPOINT_OUTPUT_DIR_PHASE2=!PROJECT_ROOT!\etp_phase2_ala\checkpoints_phase2_ala_templated"
SET "BEST_CKPT=!CHECKPOINT_OUTPUT_DIR_PHASE2!\ckpt_p2_best.pth.tar"
IF EXIST "!BEST_CKPT!" (
    SET "LOAD_CHECKPOINT=!BEST_CKPT!"
    ECHO INFO: Found best Phase 2 checkpoint
    EXIT /B 0
)

REM --- Latest Phase 2 Epoch Checkpoint ---
FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "!CHECKPOINT_OUTPUT_DIR_PHASE2!\ckpt_p2_ep*_gs*.pth.tar" 2^>nul') DO (
    SET "LOAD_CHECKPOINT=!CHECKPOINT_OUTPUT_DIR_PHASE2!\%%F"
    ECHO INFO: Found latest Phase 2 checkpoint
    EXIT /B 0
)

REM --- Latest Phase 1 Checkpoint ---
SET "CHECKPOINT_LOAD_DIR_PHASE1=!PROJECT_ROOT!\etp_phase1_reconstruction\checkpoints_phase1_rec_templated"
FOR /F "delims=" %%F IN ('dir /b /o-d /a-d "!CHECKPOINT_LOAD_DIR_PHASE1!\checkpoint_p1_end_of_epoch*_step*.pth.tar" 2^>nul') DO (
    SET "LOAD_CHECKPOINT=!CHECKPOINT_LOAD_DIR_PHASE1!\%%F"
    ECHO INFO: Found latest Phase 1 checkpoint
    EXIT /B 0
)

EXIT /B 1

:ValidateEnvironment
IF NOT EXIST "!PYTHON_EXE!" (
    ECHO ERROR: Python executable not found
    EXIT /B 1
)
IF NOT EXIST "!PROJECT_ROOT!\etp_corpus_A_deepseek_r1_embeddings.npz" (
    ECHO ERROR: Missing embeddings file A
    EXIT /B 1
)
IF NOT EXIST "!PROJECT_ROOT!\etp_corpus_B_deepseek_r1_embeddings.npz" (
    ECHO ERROR: Missing embeddings file B
    EXIT /B 1
)
IF NOT EXIST "!CHECKPOINT_OUTPUT_DIR_PHASE2!" (
    MKDIR "!CHECKPOINT_OUTPUT_DIR_PHASE2!"
)
EXIT /B 0

:BuildArguments
SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --embeddings_file_A "!PROJECT_ROOT!\etp_corpus_A_deepseek_r1_embeddings.npz""
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --embeddings_file_B "!PROJECT_ROOT!\etp_corpus_B_deepseek_r1_embeddings.npz""
IF DEFINED LOAD_CHECKPOINT SET "SCRIPT_ARGS=!SCRIPT_ARGS! --load_checkpoint "!LOAD_CHECKPOINT!""
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --checkpoint_dir "!CHECKPOINT_OUTPUT_DIR_PHASE2!""

REM Add other arguments following same pattern...
REM Example for model parameters:
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --ds_r1_embedding_dim 1536"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --wubu_initial_tangent_dim 256"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --epochs 10"

EXIT /B 0

:RunTraining
ECHO.
ECHO ==================== STARTING TRAINING ====================
ECHO Command: "!PYTHON_EXE!" "!FULL_SCRIPT_PATH!" !SCRIPT_ARGS!
ECHO.

"!PYTHON_EXE!" "!FULL_SCRIPT_PATH!" !SCRIPT_ARGS!
SET "EXIT_CODE=!ERRORLEVEL!"

IF !EXIT_CODE! EQU 0 (
    ECHO SUCCESS: Training completed
) ELSE (
    ECHO ERROR: Training failed (code !EXIT_CODE!)
)

PAUSE
EXIT /B !EXIT_CODE!