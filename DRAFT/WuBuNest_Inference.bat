@echo off
:: WuBu Poetry Structure Generator
:: ------------------------------
:: Optimized for generating poem structure patterns

echo.
echo ****************************
echo * WuBu Poetry Structure Gen *
echo ****************************
echo.

:: --- Prerequisites Check ---
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH. Please install Python 3.8+ first.
    pause
    exit /b 1
)

:: Check virtual environment
if not exist "venv\" (
    echo ERROR: Virtual environment 'venv\' not found.
    pause
    exit /b 1
)

:: --- Activate Environment ---
call venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

:: --- Default Configuration Values ---
set DEFAULT_CHECKPOINT=C:\projects\bytropix\wubunest_poem_checkpoints_v03\checkpoint_epoch_1_final_vloss4.811.pt
set DEFAULT_MAX_LENGTH=200
set DEFAULT_TEMPERATURE=0.3
set DEFAULT_DEVICE=cuda
set DEFAULT_REPETITION_PENALTY=1.3
set DEFAULT_TOP_K=20
set DEFAULT_TOP_P=0.9
set CLEANUP=--cleanup

:: --- Template Selection ---
echo Choose a poem structure template:
echo 1. Title and stanzas
echo 2. Haiku format
echo 3. Sonnet format
echo 4. Indented verse
echo 5. Custom prompt
echo.

set TEMPLATE_CHOICE=1
set /p TEMPLATE_CHOICE="Select template [1-5]: "

if "%TEMPLATE_CHOICE%"=="1" (
    set SEED_TEXT=Poem #1\n\nStanza 1, line 1\nStanza 1, line 2\nStanza 1, line 3\nStanza 1, line 4\n\n
) else if "%TEMPLATE_CHOICE%"=="2" (
    set SEED_TEXT=Haiku\n\nLine 1\nLine 2\nLine 3\n\n
) else if "%TEMPLATE_CHOICE%"=="3" (
    set SEED_TEXT=Sonnet\n\nLine 1\nLine 2\nLine 3\nLine 4\n\n
) else if "%TEMPLATE_CHOICE%"=="4" (
    set SEED_TEXT=Free Verse\n\nLine 1\n    Line 2\nLine 3\n\n
) else if "%TEMPLATE_CHOICE%"=="5" (
    set /p CUSTOM_SEED="Enter your prompt: "
    set SEED_TEXT=%CUSTOM_SEED%
) else (
    echo Invalid choice. Using default template.
    set SEED_TEXT=Poem #1\n\nStanza 1, line 1\nStanza 1, line 2\nStanza 1, line 3\nStanza 1, line 4\n\n
)

:: --- Start Generation ---
echo.
echo Starting poetry structure generation...
echo.

python structure_generator.py ^
    --checkpoint %DEFAULT_CHECKPOINT% ^
    --seed_text "%SEED_TEXT%" ^
    --max_length %DEFAULT_MAX_LENGTH% ^
    --temperature %DEFAULT_TEMPERATURE% ^
    --repetition_penalty %DEFAULT_REPETITION_PENALTY% ^
    --top_k %DEFAULT_TOP_K% ^
    --top_p %DEFAULT_TOP_P% ^
    --device %DEFAULT_DEVICE% ^
    %CLEANUP%

:: --- End ---
echo.
echo Structure generation complete.
echo.
echo Note: This model was trained on structural patterns rather than poetic content.
echo       It excels at creating poem layouts but not meaningful word content.
echo.

pause