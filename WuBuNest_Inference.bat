@echo off
:: WuBu Nesting Model Inference Launcher
:: ------------------------------------
:: Provides a user-friendly way to run inference with the WuBu Nesting model.

echo.
echo ****************************
echo * WuBu Nesting Inference *
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
    echo Please ensure the virtual environment is set up correctly in the current directory.
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
set DEFAULT_CHECKPOINT=C:/projects/bytropix/wubunest_poem_checkpoints\checkpoint_epoch_2_final_vloss5.468.pt
set DEFAULT_MAX_LENGTH=150
set DEFAULT_TEMPERATURE=0.8
set DEFAULT_DEVICE=cuda
set DEFAULT_REPETITION_PENALTY=1.1
set DEFAULT_TOP_K=0
set DEFAULT_TOP_P=0.0


:: --- User Configuration Choice ---
set USE_CUSTOM_SETTINGS=0
set /p USE_DEFAULTS="Use default settings? (Y/n): "
if /i "%USE_DEFAULTS%"=="n" set USE_CUSTOM_SETTINGS=1

:: Jump to appropriate configuration section
if %USE_CUSTOM_SETTINGS% equ 1 goto configure_custom_settings
goto configure_default_settings

:: --- Default Settings Block ---
:configure_default_settings
echo Using default settings.
set CHECKPOINT_PATH="%DEFAULT_CHECKPOINT%"
set MAX_LENGTH=%DEFAULT_MAX_LENGTH%
set TEMPERATURE=%DEFAULT_TEMPERATURE%
set DEVICE=%DEFAULT_DEVICE%
set REPETITION_PENALTY=%DEFAULT_REPETITION_PENALTY%
set TOP_K=%DEFAULT_TOP_K%
set TOP_P=%DEFAULT_TOP_P%
goto display_settings

:: --- Custom Settings Block ---
:configure_custom_settings
echo.
echo Please configure your inference settings:
echo.

:checkpoint_prompt
set CHECKPOINT_PATH=
set /p CHECKPOINT_PATH="Enter path to model checkpoint [%DEFAULT_CHECKPOINT%]: "
if "%CHECKPOINT_PATH%"=="" set CHECKPOINT_PATH=%DEFAULT_CHECKPOINT%
set CHECKPOINT_PATH="%CHECKPOINT_PATH%"
if not exist %CHECKPOINT_PATH% (
    echo ERROR: Checkpoint file not found at: %CHECKPOINT_PATH%
    set CHECKPOINT_PATH=
    goto checkpoint_prompt
)

:max_length_prompt
set MAX_LENGTH=
set /p MAX_LENGTH="Max new bytes per turn [%DEFAULT_MAX_LENGTH%]: "
if "%MAX_LENGTH%"=="" set MAX_LENGTH=%DEFAULT_MAX_LENGTH%
echo %MAX_LENGTH% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Please enter a positive integer. & goto max_length_prompt )

:temperature_prompt
set TEMPERATURE=
set /p TEMPERATURE="Sampling temperature (e.g., 0.8) [%DEFAULT_TEMPERATURE%]: "
if "%TEMPERATURE%"=="" set TEMPERATURE=%DEFAULT_TEMPERATURE%
echo %TEMPERATURE% | findstr /r /c:"^[0-9]*\.[0-9]+$" /c:"^[0-9]+$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Please enter a valid non-negative number. & goto temperature_prompt )

:device_prompt
set DEVICE=
set /p DEVICE="Device (cuda/cpu) [%DEFAULT_DEVICE%]: "
if "%DEVICE%"=="" set DEVICE=%DEFAULT_DEVICE%
if /i not "%DEVICE%"=="cuda" if /i not "%DEVICE%"=="cpu" (
    echo ERROR: Please enter 'cuda' or 'cpu'.
    goto device_prompt
)

:repetition_penalty_prompt
set REPETITION_PENALTY=
set /p REPETITION_PENALTY="Repetition penalty (e.g., 1.1) [%DEFAULT_REPETITION_PENALTY%]: "
if "%REPETITION_PENALTY%"=="" set REPETITION_PENALTY=%DEFAULT_REPETITION_PENALTY%
echo %REPETITION_PENALTY% | findstr /r /c:"^[0-9]*\.[0-9]+$" /c:"^[0-9]+$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Please enter a valid non-negative number. & goto repetition_penalty_prompt )

:top_k_prompt
set TOP_K=
set /p TOP_K="Top-K value (0 to disable) [%DEFAULT_TOP_K%]: "
if "%TOP_K%"=="" set TOP_K=%DEFAULT_TOP_K%
echo %TOP_K% | findstr /r "^[0-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Please enter a non-negative integer. & goto top_k_prompt )

:top_p_prompt
set TOP_P=
set /p TOP_P="Top-P (nucleus) value (0.0 to disable) [%DEFAULT_TOP_P%]: "
if "%TOP_P%"=="" set TOP_P=%DEFAULT_TOP_P%
echo %TOP_P% | findstr /r /c:"^[0-1]\.[0-9]+$" /c:"^0\.[0-9]+$" /c:"^0$" /c:"^1$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Please enter a value between 0 and 1. & goto top_p_prompt )


goto display_settings

:: --- Display Settings ---
:display_settings
echo.
echo Configuration Summary:
echo ---------------------
echo Checkpoint Path: %CHECKPOINT_PATH%
echo Max Length: %MAX_LENGTH%
echo Temperature: %TEMPERATURE%
echo Device: %DEVICE%
echo Repetition Penalty: %REPETITION_PENALTY%
echo Top-K: %TOP_K%
echo Top-P: %TOP_P%
echo.

:: --- Start Inference ---
:start_inference
echo.
echo Starting WuBu Nesting inference session...
echo Press Ctrl+C at any time to exit the script.
echo.

python WuBuNest_Inference.py ^
    --checkpoint_path %CHECKPOINT_PATH% ^
    --seed_text "%SEED_TEXT%" ^
    --max_length %MAX_LENGTH% ^
    --temperature %TEMPERATURE% ^
    --device %DEVICE% ^
    --repetition_penalty %REPETITION_PENALTY% ^
    --top_k %TOP_K% ^
    --top_p %TOP_P%
pause
:: --- Cleanup ---
:cleanup
echo.
if %ERRORLEVEL% equ 0 (
    echo Inference session completed successfully.
) else (
    echo ERROR: Inference session ended with an error (code %ERRORLEVEL%). Check the output above.
)

pause
exit /b %ERRORLEVEL%
