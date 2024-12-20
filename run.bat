@echo off
echo Activating virtual environment and running main.py...

REM Check if the virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Run the main.py script
python main.py

REM Inform the user that the script has completed
echo main.py execution completed.
pause
