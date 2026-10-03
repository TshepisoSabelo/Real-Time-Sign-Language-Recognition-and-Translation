@echo off
REM Activate virtual environment and run Python scripts
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist .venv\ (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

REM Activate virtual environment
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

echo.
echo Virtual environment setup complete!
echo To activate the virtual environment manually, run: .venv\Scripts\activate.bat
echo To deactivate, run: deactivate
pause
