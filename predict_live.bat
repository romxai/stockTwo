@echo off
REM ============================================================
REM Live Stock Prediction Runner
REM ============================================================
REM Usage: predict_live.bat TICKER [DAYS]
REM Example: predict_live.bat AAPL 7
REM
REM This script runs live predictions using trained models.
REM Results are saved to the results/TICKER folder.
REM ============================================================

setlocal enabledelayedexpansion

REM Check if ticker is provided
if "%~1"=="" (
    echo.
    echo Error: Please provide a ticker symbol
    echo.
    echo Usage: predict_live.bat TICKER [DAYS]
    echo Example: predict_live.bat AAPL 7
    echo.
    exit /b 1
)

set TICKER=%~1
set DAYS=%~2

REM Default to 7 days if not specified
if "%DAYS%"=="" set DAYS=7

REM Set results folder based on ticker (handles common variations)
set RESULTS_FOLDER=results\%TICKER%

REM Try common variations (APPL vs AAPL, etc)
if not exist "%RESULTS_FOLDER%" (
    if /I "%TICKER%"=="AAPL" set RESULTS_FOLDER=results\APPL
    if /I "%TICKER%"=="APPL" set RESULTS_FOLDER=results\APPL
)

REM Create results folder if it doesn't exist
if not exist "%RESULTS_FOLDER%" (
    mkdir "%RESULTS_FOLDER%"
    echo Created results folder: %RESULTS_FOLDER%
)

REM Suppress TensorFlow warnings
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the prediction script
echo.
echo ============================================================
echo Live Stock Prediction
echo ============================================================
echo Ticker: %TICKER%
echo Days: %DAYS%
echo Output: %RESULTS_FOLDER%
echo ============================================================
echo.

python venv\share\predict_live.py --model_folder=%RESULTS_FOLDER% --ticker=%TICKER% --days=%DAYS% --no-plot 2>nul

echo.
echo ============================================================
echo Completed! Results saved to: %RESULTS_FOLDER%
echo ============================================================
echo.

endlocal
