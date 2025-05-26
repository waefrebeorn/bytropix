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
SET "SCRIPT_NAME=etp_embedding_extractor_enhanced.py"
SET "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%"

REM =====================================================================
REM Configuration
REM =====================================================================
SET "MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SET "TEXTS_FILE_A=C:\Projects\bytropix\draftPY\mteb_corpus_A.txt"
SET "TEXTS_FILE_B=C:\Projects\bytropix\draftPY\mteb_corpus_B.txt"
SET "NUM_DUMMY_TEXTS_A=70"
SET "NUM_DUMMY_TEXTS_B=60"
SET "DUMMY_TEXT_USE_DIVERSE_POOL=true"
SET "DUMMY_TEXT_MIN_WORDS=5"
SET "DUMMY_TEXT_MAX_WORDS=30"
SET "OUTPUT_DIR=%PROJECT_ROOT%"
SET "OUTPUT_FILENAME_A=etp_corpus_A_deepseek_r1_embeddings"
SET "OUTPUT_FILENAME_B=etp_corpus_B_deepseek_r1_embeddings"
SET "OUTPUT_FORMAT=npz"
SET "OUTPUT_PATH_A=%OUTPUT_DIR%\%OUTPUT_FILENAME_A%.%OUTPUT_FORMAT%"
SET "OUTPUT_PATH_B=%OUTPUT_DIR%\%OUTPUT_FILENAME_B%.%OUTPUT_FORMAT%"
SET "DEVICE=auto"
SET "BATCH_SIZE=8"
SET "MAX_LENGTH=512"
SET "POOLING_STRATEGY=mean"
SET "TRUST_REMOTE_CODE=true"
SET "USE_BERT_TINY_FOR_TEST=false"

REM =====================================================================
REM Script Argument Assembly
REM =====================================================================
SET "SCRIPT_ARGS="
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --model_name_or_path "%MODEL_NAME%""
IF DEFINED TEXTS_FILE_A (
    IF NOT "%TEXTS_FILE_A%"=="" ( SET "SCRIPT_ARGS=!SCRIPT_ARGS! --texts_file_A "%TEXTS_FILE_A%""
    ) ELSE ( SET "SCRIPT_ARGS=!SCRIPT_ARGS! --num_dummy_texts_A %NUM_DUMMY_TEXTS_A%" )
) ELSE ( SET "SCRIPT_ARGS=!SCRIPT_ARGS! --num_dummy_texts_A %NUM_DUMMY_TEXTS_A%" )
IF DEFINED TEXTS_FILE_B (
     IF NOT "%TEXTS_FILE_B%"=="" ( SET "SCRIPT_ARGS=!SCRIPT_ARGS! --texts_file_B "%TEXTS_FILE_B%"" )
)
IF /I "%DUMMY_TEXT_USE_DIVERSE_POOL%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dummy_text_use_diverse_pool true"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dummy_text_min_words %DUMMY_TEXT_MIN_WORDS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --dummy_text_max_words %DUMMY_TEXT_MAX_WORDS%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --output_path_A "%OUTPUT_PATH_A%""
IF DEFINED OUTPUT_PATH_B (
    IF NOT "%OUTPUT_PATH_B%"=="" (
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --output_path_B "%OUTPUT_PATH_B%""
        SET "SCRIPT_ARGS=!SCRIPT_ARGS! --num_dummy_texts_B %NUM_DUMMY_TEXTS_B%"
    )
)
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --output_format %OUTPUT_FORMAT%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --device %DEVICE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --batch_size %BATCH_SIZE%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --max_length %MAX_LENGTH%"
SET "SCRIPT_ARGS=!SCRIPT_ARGS! --pooling_strategy %POOLING_STRATEGY%"
IF /I "%TRUST_REMOTE_CODE%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --trust_remote_code true"
IF /I "%USE_BERT_TINY_FOR_TEST%"=="true" SET "SCRIPT_ARGS=!SCRIPT_ARGS! --use_bert_tiny_for_test"

REM =====================================================================
REM Pre-Run Echo
REM =====================================================================
ECHO ======================================================
ECHO ETP Enhanced Embedding Extractor
ECHO Python: %PYTHON_EXE%
ECHO Script Path: %FULL_SCRIPT_PATH%
ECHO Model: %MODEL_NAME%
ECHO Output A: %OUTPUT_PATH_A%
IF DEFINED OUTPUT_PATH_B ( IF NOT "%OUTPUT_PATH_B%"=="" ECHO Output B: %OUTPUT_PATH_B% )
ECHO Device: %DEVICE%
ECHO Pooling: %POOLING_STRATEGY%
ECHO Dummy Texts Diverse Pool: %DUMMY_TEXT_USE_DIVERSE_POOL%
ECHO ======================================================
ECHO.

REM =====================================================================
REM Environment Setup and Execution
REM =====================================================================
IF NOT EXIST "%PYTHON_EXE%" (
    ECHO ERROR: Python executable not found at %PYTHON_EXE%
    GOTO :End
)

IF NOT EXIST "%OUTPUT_DIR%" (
    MKDIR "%OUTPUT_DIR%"
    IF ERRORLEVEL 1 (
        ECHO ERROR: Failed to create output directory "%OUTPUT_DIR%". Check permissions.
        GOTO :End
    )
)
ECHO.

ECHO Starting embedding extraction script: %SCRIPT_NAME%
ECHO Full command: %PYTHON_EXE% "%FULL_SCRIPT_PATH%" !SCRIPT_ARGS!
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