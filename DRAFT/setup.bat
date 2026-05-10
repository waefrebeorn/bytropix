@echo off
echo Setting up the Bytropix environment and installing dependencies...

REM Create a virtual environment if it doesn't already exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Upgrade pip to the latest version
echo Upgrading pip...
pip install --upgrade pip

REM Check for CUDA version using nvcc
echo Checking for CUDA support using nvcc...
nvcc --version > cuda_check.txt 2>nul

REM Check if nvcc command was successful
if %errorlevel% NEQ 0 (
    echo CUDA not found. Setting CUDA version to CPU.
    set "cuda_version=CPU"
) else (
    REM Read the CUDA version from nvcc output
    set "cuda_version=CPU"
    for /f "tokens=5 delims= " %%A in ('findstr /r /c:"release" cuda_check.txt') do (
        set "cuda_version=%%A"
    )
    
    REM Extract only the major and minor version (e.g., 12.4) and remove any trailing commas or extra characters
    for /f "tokens=1,2 delims=." %%A in ("%cuda_version:,=%") do (
        set "cuda_version_major=%%A"
        set "cuda_version_minor=%%B"
    )
    set "cuda_version=%cuda_version_major%.%cuda_version_minor%"
)

REM Display detected CUDA information
echo CUDA Version Detected: %cuda_version%

REM Create or overwrite the requirements.txt file
echo Creating requirements.txt...
(
    echo torch
    echo torchvision
    echo datasets
    echo transformers
    echo tokenizers
    echo torchtext
    echo matplotlib
    echo scipy
    echo tensorboard
    echo psutil
    echo scikit-learn
	echo wandb
) > requirements.txt

REM Install the appropriate version of PyTorch based on CUDA availability
if /i "%cuda_version%" == "CPU" (
    echo No CUDA device detected or CUDA not available. Installing the CPU version of PyTorch...
    pip install torch torchvision
) else (
    echo CUDA detected. Installing the CUDA version of PyTorch for version %cuda_version%...
    if "%cuda_version%" == "12.4" (
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ) else (
        echo Unrecognized or unsupported CUDA version detected. Proceeding with default CUDA installation...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    )
)

REM Cleanup the CUDA check file
if exist cuda_check.txt del cuda_check.txt

REM Install other necessary libraries from requirements.txt
echo Installing other dependencies from requirements.txt...
pip install -r requirements.txt

REM Download the WikiText-2 dataset using Hugging Face's datasets library
echo Downloading WikiText-2 dataset...
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').to_pandas().to_csv('./data/wikitext_train.csv'); load_dataset('wikitext', 'wikitext-2-raw-v1', split='test').to_pandas().to_csv('./data/wikitext_test.csv')"

REM Inform the user that setup is complete
echo Setup complete. You can now run the main.py script.
pause
