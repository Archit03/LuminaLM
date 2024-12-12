@echo off
setlocal enabledelayedexpansion

:: Set environment variables
set PYTHON_PATH=python
set SCRIPT_PATH=tokenizer.py
set DATA_PATH=Data
set CACHE_DIR=.cache
set LOG_FILE=tokenizer.log
set VOCAB_SIZE=60000
set MIN_FREQ=2
set CHUNK_SIZE=10000

:: Check Python installation
%PYTHON_PATH% --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please ensure Python is installed and in PATH.
    exit /b 1
)

:: Create directories if they don't exist
if not exist "%DATA_PATH%" mkdir "%DATA_PATH%"
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"

:: Check if script exists
if not exist "%SCRIPT_PATH%" (
    echo Error: Tokenizer script not found at %SCRIPT_PATH%
    exit /b 1
)

:: Run the tokenizer with enhanced configuration
echo Starting tokenizer training...
%PYTHON_PATH% "%SCRIPT_PATH%" ^
    --local_data_path "%DATA_PATH%" ^
    --vocab_size %VOCAB_SIZE% ^
    --min_frequency %MIN_FREQ% ^
    --log_file "%LOG_FILE%" ^
    --cache_dir "%CACHE_DIR%" ^
    --chunk_size %CHUNK_SIZE% ^
    --max_workers 6

if errorlevel 1 (
    echo Error: Tokenizer training failed. Check %LOG_FILE% for details.
    exit /b 1
) else (
    echo Tokenizer training completed successfully.
)

endlocal 