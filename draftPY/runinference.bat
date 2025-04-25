@echo off
:: Bytropix Interactive Inference Launcher (Restructured)
:: ----------------------------------------------------
:: Provides a user-friendly way to launch the interactive inference mode
:: using flags and goto instead of if/else blocks.

echo.
echo *******************************
echo * Bytropix Inference Tool   *
echo *******************************
echo.

:: --- Prerequisites Check ---
:: Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH. Please install Python 3.8+ first.
    pause
    exit /b 1
)

:: Check virtual environment
if not exist "venv\" (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first to create the virtual environment.
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
:: !! IMPORTANT: Update these defaults to match your trained checkpoint !!
set DEFAULT_CHECKPOINT=C:\projects\bytropix\checkpoints\checkpoint_final.pt
set DEFAULT_MAX_LENGTH=150
set DEFAULT_TEMPERATURE=0.75
set DEFAULT_LOW_ENTROPY=0.3
set DEFAULT_MEDIUM_ENTROPY=1.2
set DEFAULT_HIGH_ENTROPY=2.5
:: Model Dimension Defaults (Update these based on your trained model!)
set DEFAULT_LOCAL_HIDDEN_SIZE=512
set DEFAULT_GLOBAL_HIDDEN_SIZE=1024
set DEFAULT_NUM_LOCAL_ENCODER_LAYERS=2
set DEFAULT_NUM_GLOBAL_LAYERS=16
set DEFAULT_NUM_LOCAL_DECODER_LAYERS=6
set DEFAULT_WINDOW_SIZE=256

:: --- User Configuration Choice ---
set USE_CUSTOM_SETTINGS=0
set /p USE_DEFAULTS="Use default settings (including model dimensions)? (Y/n): "
if /i "%USE_DEFAULTS%"=="n" set USE_CUSTOM_SETTINGS=1

:: Jump to appropriate configuration section
if %USE_CUSTOM_SETTINGS% equ 1 goto configure_custom_settings
goto configure_default_settings


:: --- Default Settings Block ---
:configure_default_settings
echo DEBUG: Setting defaults...
set CHECKPOINT_PATH="%DEFAULT_CHECKPOINT%"
set MAX_LENGTH=%DEFAULT_MAX_LENGTH%
set TEMPERATURE=%DEFAULT_TEMPERATURE%
set LOW_ENTROPY=%DEFAULT_LOW_ENTROPY%
set MEDIUM_ENTROPY=%DEFAULT_MEDIUM_ENTROPY%
set HIGH_ENTROPY=%DEFAULT_HIGH_ENTROPY%
set LOCAL_HIDDEN_SIZE=%DEFAULT_LOCAL_HIDDEN_SIZE%
set GLOBAL_HIDDEN_SIZE=%DEFAULT_GLOBAL_HIDDEN_SIZE%
set NUM_LOCAL_ENCODER_LAYERS=%DEFAULT_NUM_LOCAL_ENCODER_LAYERS%
set NUM_GLOBAL_LAYERS=%DEFAULT_NUM_GLOBAL_LAYERS%
set NUM_LOCAL_DECODER_LAYERS=%DEFAULT_NUM_LOCAL_DECODER_LAYERS%
set WINDOW_SIZE=%DEFAULT_WINDOW_SIZE%
echo.
echo Using default settings:
goto display_settings


:: --- Custom Settings Block ---
:configure_custom_settings
echo.
echo Please configure your inference settings:
echo.

:: Checkpoint path
:checkpoint_prompt
set CHECKPOINT_PATH=
set /p CHECKPOINT_PATH="Enter path to model checkpoint [%DEFAULT_CHECKPOINT%]: "
if "%CHECKPOINT_PATH%"=="" set CHECKPOINT_PATH=%DEFAULT_CHECKPOINT%
:: Quote the path when setting it, in case it contains spaces later
set CHECKPOINT_PATH="%CHECKPOINT_PATH%"
if not exist %CHECKPOINT_PATH% (
    echo ERROR: Checkpoint file not found at: %CHECKPOINT_PATH%
    goto checkpoint_prompt
)

:: Generation parameters
:length_prompt
set MAX_LENGTH=
set /p MAX_LENGTH="Max new bytes per turn [%DEFAULT_MAX_LENGTH%]: "
if "%MAX_LENGTH%"=="" set MAX_LENGTH=%DEFAULT_MAX_LENGTH%
echo %MAX_LENGTH% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Please enter a positive integer.
    goto length_prompt
)

:temp_prompt
set TEMPERATURE=
set /p TEMPERATURE="Sampling temperature (e.g., 0.75) [%DEFAULT_TEMPERATURE%]: "
if "%TEMPERATURE%"=="" set TEMPERATURE=%DEFAULT_TEMPERATURE%
if "%TEMPERATURE%"=="" (
    echo ERROR: Temperature cannot be empty.
    goto temp_prompt
)

:: Model Dimension Parameters
echo.
echo Configure Model Dimensions (match the trained checkpoint):

:local_hidden_prompt
set LOCAL_HIDDEN_SIZE=
set /p LOCAL_HIDDEN_SIZE="Local Hidden Size [%DEFAULT_LOCAL_HIDDEN_SIZE%]: "
if "%LOCAL_HIDDEN_SIZE%"=="" set LOCAL_HIDDEN_SIZE=%DEFAULT_LOCAL_HIDDEN_SIZE%
echo %LOCAL_HIDDEN_SIZE% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto local_hidden_prompt )

:global_hidden_prompt
set GLOBAL_HIDDEN_SIZE=
set /p GLOBAL_HIDDEN_SIZE="Global Hidden Size [%DEFAULT_GLOBAL_HIDDEN_SIZE%]: "
if "%GLOBAL_HIDDEN_SIZE%"=="" set GLOBAL_HIDDEN_SIZE=%DEFAULT_GLOBAL_HIDDEN_SIZE%
echo %GLOBAL_HIDDEN_SIZE% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto global_hidden_prompt )

:local_enc_layers_prompt
set NUM_LOCAL_ENCODER_LAYERS=
set /p NUM_LOCAL_ENCODER_LAYERS="Num Local Encoder Layers [%DEFAULT_NUM_LOCAL_ENCODER_LAYERS%]: "
if "%NUM_LOCAL_ENCODER_LAYERS%"=="" set NUM_LOCAL_ENCODER_LAYERS=%DEFAULT_NUM_LOCAL_ENCODER_LAYERS%
echo %NUM_LOCAL_ENCODER_LAYERS% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto local_enc_layers_prompt )

:global_layers_prompt
set NUM_GLOBAL_LAYERS=
set /p NUM_GLOBAL_LAYERS="Num Global Layers [%DEFAULT_NUM_GLOBAL_LAYERS%]: "
if "%NUM_GLOBAL_LAYERS%"=="" set NUM_GLOBAL_LAYERS=%DEFAULT_NUM_GLOBAL_LAYERS%
echo %NUM_GLOBAL_LAYERS% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto global_layers_prompt )

:local_dec_layers_prompt
set NUM_LOCAL_DECODER_LAYERS=
set /p NUM_LOCAL_DECODER_LAYERS="Num Local Decoder Layers [%DEFAULT_NUM_LOCAL_DECODER_LAYERS%]: "
if "%NUM_LOCAL_DECODER_LAYERS%"=="" set NUM_LOCAL_DECODER_LAYERS=%DEFAULT_NUM_LOCAL_DECODER_LAYERS%
echo %NUM_LOCAL_DECODER_LAYERS% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto local_dec_layers_prompt )

:window_size_prompt
set WINDOW_SIZE=
set /p WINDOW_SIZE="Window Size [%DEFAULT_WINDOW_SIZE%]: "
if "%WINDOW_SIZE%"=="" set WINDOW_SIZE=%DEFAULT_WINDOW_SIZE%
echo %WINDOW_SIZE% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto window_size_prompt )

:: Set entropy thresholds (can also be prompted if needed)
set LOW_ENTROPY=%DEFAULT_LOW_ENTROPY%
set MEDIUM_ENTROPY=%DEFAULT_MEDIUM_ENTROPY%
set HIGH_ENTROPY=%DEFAULT_HIGH_ENTROPY%

echo.
echo Using custom settings:
goto display_settings


:: --- Display Settings ---
:display_settings
echo.
echo  Configuration Summary:
echo  ---------------------
echo  Checkpoint: %CHECKPOINT_PATH%
echo  Max Length: %MAX_LENGTH%
echo  Temperature: %TEMPERATURE%
echo  Low Entropy Threshold: %LOW_ENTROPY%
echo  Medium Entropy Threshold: %MEDIUM_ENTROPY%
echo  High Entropy Threshold: %HIGH_ENTROPY%
echo.
echo  Model Dimensions:
echo  Local Hidden Size: %LOCAL_HIDDEN_SIZE%
echo  Global Hidden Size: %GLOBAL_HIDDEN_SIZE%
echo  Local Enc Layers: %NUM_LOCAL_ENCODER_LAYERS%
echo  Global Layers: %NUM_GLOBAL_LAYERS%
echo  Local Dec Layers: %NUM_LOCAL_DECODER_LAYERS%
echo  Window Size: %WINDOW_SIZE%
echo.


:: --- Device Selection ---
:device_selection
set DEVICE_FLAG=
set /p USE_CPU="Force CPU mode (skip GPU detection)? (y/N): "
if /i "%USE_CPU%"=="y" (
    set DEVICE_FLAG=--cpu
    echo.
    echo WARNING: Forcing CPU mode - performance will be slower.
) else (
    set DEVICE_FLAG=
)
echo.


:: --- Start Inference ---
:start_inference
echo.
echo Starting interactive inference session...
echo Press Ctrl+C at any time to exit
echo.


python inference.py interactive ^
  --checkpoint_path %CHECKPOINT_PATH% ^
  --max_length %MAX_LENGTH% ^
  --temperature %TEMPERATURE% ^
  --low_entropy %LOW_ENTROPY% ^
  --medium_entropy %MEDIUM_ENTROPY% ^
  --high_entropy %HIGH_ENTROPY% ^
  --local_hidden_size %LOCAL_HIDDEN_SIZE% ^
  --global_hidden_size %GLOBAL_HIDDEN_SIZE% ^
  --num_local_encoder_layers %NUM_LOCAL_ENCODER_LAYERS% ^
  --num_global_layers %NUM_GLOBAL_LAYERS% ^
  --num_local_decoder_layers %NUM_LOCAL_DECODER_LAYERS% ^
  --window_size %WINDOW_SIZE% ^
  %DEVICE_FLAG%

:: --- Cleanup ---
:cleanup
echo.
if %ERRORLEVEL% equ 0 (
    echo Inference session completed successfully.
) else (
    echo ERROR: Inference session ended with an error (code %ERRORLEVEL%)
)

pause
exit /b %ERRORLEVEL%
