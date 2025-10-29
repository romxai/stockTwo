@echo off
REM ============================================================================
REM Setup Virtual Environment and Install Requirements
REM This script creates a Python virtual environment and installs all dependencies
REM ============================================================================

echo.
echo ============================================================================
echo STOCK PREDICTION - ENVIRONMENT SETUP
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.10 or later from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

REM Check Python version (requires 3.9+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

echo.
echo [2/4] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo Virtual environment created successfully!

echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/4] Installing requirements (this may take 5-10 minutes)...
echo Installing PyTorch with CUDA 12.6 support...
python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install requirements!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo SETUP COMPLETE!
echo ============================================================================
echo.
echo Virtual environment created: venv\
echo All dependencies installed successfully!
echo.
echo To activate the environment manually, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the full pipeline, use:
echo   run_pipeline.bat
echo.
echo ============================================================================

pause
