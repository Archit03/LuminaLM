@echo off
setlocal enabledelayedexpansion

:: Set environment variables
set PYTHON_PATH=python
set SCRIPT_PATH=tokenizer.py
set LOG_FILE=tokenizer.log
set MIN_FREQ=2
set CHUNK_SIZE=10000

:: Check Python installation
%PYTHON_PATH% --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please ensure Python is installed and in PATH.
    exit /b 1
)

:: Check if script exists
if not exist "%SCRIPT_PATH%" (
    echo Error: Tokenizer script not found at %SCRIPT_PATH%
    exit /b 1
)

:: Check CUDA availability and echo status
echo Checking hardware configuration...
%PYTHON_PATH% -c "import torch; print('I\'m using ' + ('CUDA' if torch.cuda.is_available() else 'CPU') + ' to do my job')"
echo.

:: Run the tokenizer with enhanced configuration
echo Starting tokenizer training...
%PYTHON_PATH% "%SCRIPT_PATH%" ^
    --min_frequency %MIN_FREQ% ^
    --log_file "%LOG_FILE%" ^
    --chunk_size %CHUNK_SIZE% ^
    --max_workers 8

if errorlevel 1 (
    echo Error: Tokenizer training failed. Check %LOG_FILE% for details.
    exit /b 1
) else (
    echo Tokenizer training completed successfully.
)

endlocal 