@echo off
REM ============================================================================
REM Run Stock Prediction Pipeline
REM This script activates the virtual environment and runs the main pipeline
REM ============================================================================

echo.
echo ============================================================================
echo STOCK PREDICTION - RUNNING PIPELINE
echo ============================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_venv.bat first to create the environment.
    pause
    exit /b 1
)

REM Check if data file exists
if not exist "data\AAPL.csv" (
    echo.
    echo WARNING: Stock data file not found at data\AAPL.csv
    echo The pipeline will create synthetic news data for demonstration.
    echo.
    echo For real data analysis, please place your AAPL.csv file in the data\ folder.
    echo.
    pause
)

echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [2/2] Running stock prediction pipeline...
echo.
echo ============================================================================
echo.

cd src
python main.py

if %errorlevel% neq 0 (
    echo.
    echo ============================================================================
    echo ERROR: Pipeline execution failed!
    echo ============================================================================
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo ============================================================================
echo PIPELINE COMPLETE!
echo ============================================================================
echo.
echo Results saved in:
echo   - models\best_model.pt (best checkpoint)
echo   - models\stock_predictor_complete.pkl (full package)
echo.
echo Check the logs\ folder for training logs and visualizations.
echo.
echo ============================================================================

pause
