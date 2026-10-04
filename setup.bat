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

REM Start the frontend in a separate command window.
if not exist "%~dp0frontend\SignAI\package.json" (
    echo Frontend package file not found: frontend\SignAI\package.json
    exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo npm was not found. Install Node.js and npm to start the frontend.
    exit /b 1
)

pushd "%~dp0frontend\SignAI"
if not exist node_modules\ (
    echo Installing frontend dependencies...
    call npm install
    if errorlevel 1 (
        echo Failed to install frontend dependencies.
        popd
        exit /b 1
    )
)
echo Starting the SignAI frontend at http://127.0.0.1:5173 ...
start "SignAI Frontend" cmd /k npm run dev
popd

echo Starting the FastAPI server at http://127.0.0.1:8000 ...
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo Uvicorn server exited with an error.
    exit /b 1
)
