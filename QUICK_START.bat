@echo off
REM ============================================================================
REM Quick Start Guide
REM ============================================================================

echo.
echo ============================================================================
echo STOCK PREDICTION - QUICK START GUIDE
echo ============================================================================
echo.
echo This wizard will help you set up and run the stock prediction pipeline.
echo.
echo ============================================================================
echo.

REM Step 1: Check Python
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ Python is not installed!
    echo.
    echo Please install Python 3.10 or later from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

python --version
echo ✅ Python found!
echo.

REM Step 2: Check data file
echo Step 2: Checking data file...
if exist "data\AAPL.csv" (
    echo ✅ Data file found: data\AAPL.csv
    echo.
) else (
    echo ⚠️  Warning: data\AAPL.csv not found!
    echo.
    echo The pipeline will use synthetic data for demonstration.
    echo For real predictions, place your CSV file in the data\ folder.
    echo.
    echo See data\README.md for CSV format requirements.
    echo.
)

REM Step 3: Check environment
echo Step 3: Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo ✅ Virtual environment already exists!
    echo.
    
    REM Verify installation
    echo Verifying installation...
    call venv\Scripts\activate.bat
    python scripts\verify_installation.py
    
    if %errorlevel% neq 0 (
        echo.
        echo ❌ Installation verification failed!
        echo Please run setup_venv.bat to reinstall.
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo ============================================================================
    echo READY TO RUN!
    echo ============================================================================
    echo.
    echo Your environment is set up and ready!
    echo.
    echo To run the full pipeline, execute:
    echo   run_pipeline.bat
    echo.
    echo This will:
    echo   1. Load and preprocess your data
    echo   2. Extract FinBERT embeddings
    echo   3. Train the model (30-45 minutes)
    echo   4. Evaluate performance
    echo   5. Save the trained model
    echo.
    echo Expected accuracy: 80%+
    echo.
    echo ============================================================================
    
) else (
    echo ❌ Virtual environment not found!
    echo.
    echo Please run setup_venv.bat first to:
    echo   1. Create virtual environment
    echo   2. Install PyTorch with CUDA support
    echo   3. Install all dependencies
    echo.
    echo This takes about 5-10 minutes.
    echo.
    echo Run setup_venv.bat now? (Y/N)
    
    set /p response="> "
    
    if /i "%response%"=="Y" (
        echo.
        echo Running setup_venv.bat...
        call setup_venv.bat
        
        if %errorlevel% neq 0 (
            echo.
            echo ❌ Setup failed!
            pause
            exit /b 1
        )
        
        echo.
        echo ============================================================================
        echo SETUP COMPLETE!
        echo ============================================================================
        echo.
        echo Environment is ready! Run this script again or execute:
        echo   run_pipeline.bat
        echo.
    ) else (
        echo.
        echo Please run setup_venv.bat when ready.
        echo.
    )
)

pause
