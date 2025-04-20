@echo off
:: SFIN Interactive Inference Launcher (Restructured & Synced with Training Params - v4)
:: -----------------------------------------------------------------------------
:: Provides a user-friendly way to launch the interactive inference mode
:: for the SFIN model (using sfin_inference.py).
:: Default model parameters are synced with runsfin.bat example provided.
:: Fixes issue with REM comments being included in variable values.

echo.
echo ****************************
echo * SFIN Inference Tool      *
echo ****************************
echo.

:: --- Prerequisites Check ---
:: Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH. Please install Python 3.8+ first.
    pause
    exit /b 1
)

:: Check virtual environment (Adjust venv path if needed)
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
:: !! IMPORTANT: Checkpoint path MUST be updated !!
set DEFAULT_CHECKPOINT=C:/projects/bytropix/checkpoints_v2\checkpoint_epoch_0_step_696.pt
REM PLEASE UPDATE THE CHECKPOINT PATH ABOVE
:: Generation Defaults
set DEFAULT_MAX_LENGTH=150
set DEFAULT_TEMPERATURE=0.8
set DEFAULT_LOW_ENTROPY=0.3
set DEFAULT_MEDIUM_ENTROPY=1.2
set DEFAULT_HIGH_ENTROPY=2.5
:: SFIN Specific Model Dimension Defaults (Synced with runsfin.bat example)
set DEFAULT_LOCAL_HIDDEN_SIZE=512
set DEFAULT_COMPLEX_DIM=512
set DEFAULT_NUM_COMPLEX_LAYERS=12
set DEFAULT_NUM_COMPLEX_HEADS=8
set DEFAULT_DECODER_MEMORY_DIM=768
set DEFAULT_CONTEXT_WINDOW=256
set DEFAULT_N_GRAM_SIZES=3 4
REM Space-separated for nargs='+'
set DEFAULT_N_GRAM_VOCAB_SIZE=30000
REM Default from sfin_main.py
set DEFAULT_NO_ENTANGLEMENT=false
REM runsfin.bat did not disable it
set DEFAULT_NO_ROPE=false
REM runsfin.bat did not disable it
set DEFAULT_PROJECTION_METHOD=concat
REM Default from sfin_main.py
:: Inference-specific Defaults
set DEFAULT_DROPOUT=0.0
set DEFAULT_SFIN_NOISE_SCALE=0.0

:: --- User Configuration Choice ---
set USE_CUSTOM_SETTINGS=0
set /p USE_DEFAULTS="Use default settings (including model dimensions from training script)? (Y/n): "
if /i "%USE_DEFAULTS%"=="n" set USE_CUSTOM_SETTINGS=1

:: Jump to appropriate configuration section
if %USE_CUSTOM_SETTINGS% equ 1 goto configure_custom_settings
goto configure_default_settings


:: --- Default Settings Block ---
:configure_default_settings
echo DEBUG: Setting defaults based on training script...
set CHECKPOINT_PATH="%DEFAULT_CHECKPOINT%"
REM Quote the path variable value itself
set MAX_LENGTH=%DEFAULT_MAX_LENGTH%
set TEMPERATURE=%DEFAULT_TEMPERATURE%
set LOW_ENTROPY=%DEFAULT_LOW_ENTROPY%
set MEDIUM_ENTROPY=%DEFAULT_MEDIUM_ENTROPY%
set HIGH_ENTROPY=%DEFAULT_HIGH_ENTROPY%
set LOCAL_HIDDEN_SIZE=%DEFAULT_LOCAL_HIDDEN_SIZE%
set COMPLEX_DIM=%DEFAULT_COMPLEX_DIM%
set NUM_COMPLEX_LAYERS=%DEFAULT_NUM_COMPLEX_LAYERS%
set NUM_COMPLEX_HEADS=%DEFAULT_NUM_COMPLEX_HEADS%
set DECODER_MEMORY_DIM=%DEFAULT_DECODER_MEMORY_DIM%
set CONTEXT_WINDOW=%DEFAULT_CONTEXT_WINDOW%
set N_GRAM_SIZES=%DEFAULT_N_GRAM_SIZES%
set N_GRAM_VOCAB_SIZE=%DEFAULT_N_GRAM_VOCAB_SIZE%
set NO_ENTANGLEMENT=%DEFAULT_NO_ENTANGLEMENT%
set NO_ROPE=%DEFAULT_NO_ROPE%
set PROJECTION_METHOD=%DEFAULT_PROJECTION_METHOD%
set DROPOUT=%DEFAULT_DROPOUT%
set SFIN_NOISE_SCALE=%DEFAULT_SFIN_NOISE_SCALE%
echo.
echo Using default settings (synced with training script defaults):
goto configure_runtime_options


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
:: Quote the path when setting it
set CHECKPOINT_PATH="%CHECKPOINT_PATH%"
if not exist %CHECKPOINT_PATH% (
    echo ERROR: Checkpoint file not found at: %CHECKPOINT_PATH%
    set CHECKPOINT_PATH=
    goto checkpoint_prompt
)

:: Generation parameters
:length_prompt
set MAX_LENGTH=
set /p MAX_LENGTH="Max new bytes per turn [%DEFAULT_MAX_LENGTH%]: "
if "%MAX_LENGTH%"=="" set MAX_LENGTH=%DEFAULT_MAX_LENGTH%
echo %MAX_LENGTH% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Please enter a positive integer. & goto length_prompt )

:temp_prompt
set TEMPERATURE=
set /p TEMPERATURE="Sampling temperature (e.g., 0.8) [%DEFAULT_TEMPERATURE%]: "
if "%TEMPERATURE%"=="" set TEMPERATURE=%DEFAULT_TEMPERATURE%
REM Basic check for number - more robust check is complex in batch
echo %TEMPERATURE% | findstr /r /c:"^[0-9]*\.[0-9][0-9]*$" /c:"^[0-9][0-9]*$" /c:"^0$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Please enter a valid non-negative number. & goto temp_prompt )

:: Model Dimension Parameters
echo.
echo Configure SFIN Model Dimensions (MUST match the trained checkpoint):

:local_hidden_prompt
set LOCAL_HIDDEN_SIZE=
set /p LOCAL_HIDDEN_SIZE="Local Hidden Size [%DEFAULT_LOCAL_HIDDEN_SIZE%]: "
if "%LOCAL_HIDDEN_SIZE%"=="" set LOCAL_HIDDEN_SIZE=%DEFAULT_LOCAL_HIDDEN_SIZE%
echo %LOCAL_HIDDEN_SIZE% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto local_hidden_prompt )

:complex_dim_prompt
set COMPLEX_DIM=
set /p COMPLEX_DIM="Complex Dimension [%DEFAULT_COMPLEX_DIM%]: "
if "%COMPLEX_DIM%"=="" set COMPLEX_DIM=%DEFAULT_COMPLEX_DIM%
echo %COMPLEX_DIM% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto complex_dim_prompt )

:complex_layers_prompt
set NUM_COMPLEX_LAYERS=
set /p NUM_COMPLEX_LAYERS="Num Complex Layers [%DEFAULT_NUM_COMPLEX_LAYERS%]: "
if "%NUM_COMPLEX_LAYERS%"=="" set NUM_COMPLEX_LAYERS=%DEFAULT_NUM_COMPLEX_LAYERS%
echo %NUM_COMPLEX_LAYERS% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto complex_layers_prompt )

:complex_heads_prompt
set NUM_COMPLEX_HEADS=
set /p NUM_COMPLEX_HEADS="Num Complex Heads [%DEFAULT_NUM_COMPLEX_HEADS%]: "
if "%NUM_COMPLEX_HEADS%"=="" set NUM_COMPLEX_HEADS=%DEFAULT_NUM_COMPLEX_HEADS%
echo %NUM_COMPLEX_HEADS% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto complex_heads_prompt )

:decoder_mem_prompt
set DECODER_MEMORY_DIM=
set /p DECODER_MEMORY_DIM="Decoder Memory Dim [%DEFAULT_DECODER_MEMORY_DIM%]: "
if "%DECODER_MEMORY_DIM%"=="" set DECODER_MEMORY_DIM=%DEFAULT_DECODER_MEMORY_DIM%
echo %DECODER_MEMORY_DIM% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto decoder_mem_prompt )

:context_window_prompt
set CONTEXT_WINDOW=
set /p CONTEXT_WINDOW="Context Window [%DEFAULT_CONTEXT_WINDOW%]: "
if "%CONTEXT_WINDOW%"=="" set CONTEXT_WINDOW=%DEFAULT_CONTEXT_WINDOW%
echo %CONTEXT_WINDOW% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto context_window_prompt )

:ngram_sizes_prompt
set N_GRAM_SIZES=
set /p N_GRAM_SIZES="N-Gram Sizes (space separated) [%DEFAULT_N_GRAM_SIZES%]: "
if "%N_GRAM_SIZES%"=="" set N_GRAM_SIZES=%DEFAULT_N_GRAM_SIZES%
if "%N_GRAM_SIZES%"=="" ( echo ERROR: N-Gram sizes cannot be empty. & goto ngram_sizes_prompt)

:ngram_vocab_prompt
set N_GRAM_VOCAB_SIZE=
set /p N_GRAM_VOCAB_SIZE="N-Gram Vocab Size [%DEFAULT_N_GRAM_VOCAB_SIZE%]: "
if "%N_GRAM_VOCAB_SIZE%"=="" set N_GRAM_VOCAB_SIZE=%DEFAULT_N_GRAM_VOCAB_SIZE%
echo %N_GRAM_VOCAB_SIZE% | findstr /r "^[1-9][0-9]*$" >nul
if %ERRORLEVEL% neq 0 ( echo ERROR: Invalid number. & goto ngram_vocab_prompt )

:proj_method_prompt
set PROJECTION_METHOD=
set /p PROJECTION_METHOD="Complex->Real Projection (concat/magnitude) [%DEFAULT_PROJECTION_METHOD%]: "
if "%PROJECTION_METHOD%"=="" set PROJECTION_METHOD=%DEFAULT_PROJECTION_METHOD%
if /i not "%PROJECTION_METHOD%"=="concat" if /i not "%PROJECTION_METHOD%"=="magnitude" (
    echo ERROR: Projection method must be 'concat' or 'magnitude'.
    goto proj_method_prompt
)

:no_entangle_prompt
set NO_ENTANGLEMENT=
set /p NO_ENTANGLEMENT_INPUT="Disable Entanglement (true/false)? [%DEFAULT_NO_ENTANGLEMENT%]: "
if "%NO_ENTANGLEMENT_INPUT%"=="" set NO_ENTANGLEMENT=%DEFAULT_NO_ENTANGLEMENT%
if /i "%NO_ENTANGLEMENT_INPUT%"=="true" set NO_ENTANGLEMENT=true
if /i "%NO_ENTANGLEMENT_INPUT%"=="false" set NO_ENTANGLEMENT=false
if "%NO_ENTANGLEMENT%"=="" ( echo ERROR: Enter 'true' or 'false'. & goto no_entangle_prompt)

:no_rope_prompt
set NO_ROPE=
set /p NO_ROPE_INPUT="Disable RoPE (true/false)? [%DEFAULT_NO_ROPE%]: "
if "%NO_ROPE_INPUT%"=="" set NO_ROPE=%DEFAULT_NO_ROPE%
if /i "%NO_ROPE_INPUT%"=="true" set NO_ROPE=true
if /i "%NO_ROPE_INPUT%"=="false" set NO_ROPE=false
if "%NO_ROPE%"=="" ( echo ERROR: Enter 'true' or 'false'. & goto no_rope_prompt)


:: Set inference-specific defaults (usually 0.0)
set LOW_ENTROPY=%DEFAULT_LOW_ENTROPY%
set MEDIUM_ENTROPY=%DEFAULT_MEDIUM_ENTROPY%
set HIGH_ENTROPY=%DEFAULT_HIGH_ENTROPY%
set DROPOUT=%DEFAULT_DROPOUT%
set SFIN_NOISE_SCALE=%DEFAULT_SFIN_NOISE_SCALE%

echo.
echo Using custom settings:
goto configure_runtime_options


:: --- Configure Runtime Flags (CPU/AMP) ---
:configure_runtime_options
echo.
:: Device Selection
:device_selection
set DEVICE_FLAG=
set USE_CPU=N REM Default to No
set /p USE_CPU="Force CPU mode (skip GPU detection)? (y/N): "
if /i "%USE_CPU%"=="y" (
    set DEVICE_FLAG=--cpu
    echo WARNING: Forcing CPU mode - performance will be slower.
)

:: AMP Selection
:amp_selection
set AMP_FLAG=
set DISABLE_AMP=N REM Default to No
set /p DISABLE_AMP="Disable Automatic Mixed Precision (AMP)? (y/N): "
if /i "%DISABLE_AMP%"=="y" (
    set AMP_FLAG=--no_amp
    echo WARNING: Disabling AMP, may increase memory usage and slow down on compatible GPUs.
)

:: Set optional boolean flags for the command based on configured vars
set ENTANGLEMENT_FLAG=
if /i "%NO_ENTANGLEMENT%"=="true" set ENTANGLEMENT_FLAG=--no_entanglement

set ROPE_FLAG=
if /i "%NO_ROPE%"=="true" set ROPE_FLAG=--no_rope

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
echo  Force CPU: %USE_CPU%
echo  Disable AMP: %DISABLE_AMP%
echo.
echo  Model Dimensions:
echo    Local Hidden Size: %LOCAL_HIDDEN_SIZE%
echo    Complex Dimension: %COMPLEX_DIM%
echo    Complex Layers: %NUM_COMPLEX_LAYERS%
echo    Complex Heads: %NUM_COMPLEX_HEADS%
echo    Decoder Memory Dim: %DECODER_MEMORY_DIM%
echo    Context Window: %CONTEXT_WINDOW%
echo    N-Gram Sizes: %N_GRAM_SIZES%
echo    N-Gram Vocab Size: %N_GRAM_VOCAB_SIZE%
echo    Projection Method: %PROJECTION_METHOD%
echo    Disable Entanglement: %NO_ENTANGLEMENT%
echo    Disable RoPE: %NO_ROPE%
echo    Dropout (Inference): %DROPOUT%
echo    SFIN Noise (Inference): %SFIN_NOISE_SCALE%
echo.


:: --- Start Inference ---
:start_inference
echo.
echo Starting SFIN interactive inference session...
echo Press Ctrl+C at any time to exit the script.
echo Type 'quit' or 'exit' in the prompt to end the session gracefully.
echo.

python sfin_inference.py interactive ^
    --checkpoint_path %CHECKPOINT_PATH% ^
    --max_length %MAX_LENGTH% ^
    --temperature %TEMPERATURE% ^
    --low_entropy %LOW_ENTROPY% ^
    --medium_entropy %MEDIUM_ENTROPY% ^
    --high_entropy %HIGH_ENTROPY% ^
    --local_hidden_size %LOCAL_HIDDEN_SIZE% ^
    --complex_dim %COMPLEX_DIM% ^
    --num_complex_layers %NUM_COMPLEX_LAYERS% ^
    --num_complex_heads %NUM_COMPLEX_HEADS% ^
    --decoder_memory_dim %DECODER_MEMORY_DIM% ^
    --context_window %CONTEXT_WINDOW% ^
    --n_gram_sizes %N_GRAM_SIZES% ^
    --n_gram_vocab_size %N_GRAM_VOCAB_SIZE% ^
    --projection_method %PROJECTION_METHOD% ^
    --dropout %DROPOUT% ^
    --sfin_noise_scale %SFIN_NOISE_SCALE% ^
    %DEVICE_FLAG% ^
    %AMP_FLAG% ^
    %ENTANGLEMENT_FLAG% ^
    %ROPE_FLAG%

:: --- Cleanup ---
:cleanup
echo.
if %ERRORLEVEL% equ 0 (
    echo SFIN Inference session completed successfully.
) else (
    echo ERROR: SFIN Inference session ended with an error (code %ERRORLEVEL%). Check logs/output above.
)

pause
exit /b %ERRORLEVEL%