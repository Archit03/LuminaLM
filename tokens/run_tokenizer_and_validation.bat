@echo off
setlocal EnableDelayedExpansion

:: Define variables
set "TOKENIZER_SCRIPT=tokenizer.py"
set "LOCAL_DATA_DIR=C:\Users\ASUS\Desktop\LuminaLM\Data"
set "VOCAB_SIZE=60000"
set "MIN_FREQUENCY=2"
set "LOG_FILE=medical_tokenization.log"
set "CACHE_DIR=C:\Users\ASUS\Desktop\LuminaLM\tokens\.data_cache"
set "PYTHON_EXECUTABLE=python" :: Replace with full path if needed (e.g., C:\Python39\python.exe)

:: Validate required directories
if not exist "%LOCAL_DATA_DIR%" (
    echo [ERROR] Data directory "%LOCAL_DATA_DIR%" does not exist. Please create it and try again.
    exit /b 1
)

if not exist "%CACHE_DIR%" (
    echo [INFO] Cache directory "%CACHE_DIR%" does not exist. Creating it...
    mkdir "%CACHE_DIR%"
)

:: Run the tokenizer script
echo [INFO] Starting tokenizer training process...
"%PYTHON_EXECUTABLE%" "%TOKENIZER_SCRIPT%" ^
    --local_data_path "%LOCAL_DATA_DIR%" ^
    --vocab_size %VOCAB_SIZE% ^
    --min_frequency %MIN_FREQUENCY% ^
    --log_file "%LOG_FILE%" ^
    --cache_dir "%CACHE_DIR%"

:: Check if the tokenizer script executed successfully
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Tokenizer script execution failed. Please check the log for details.
    exit /b 1
)

:: Verify the log file
if exist "%LOG_FILE%" (
    echo [INFO] Tokenizer process completed successfully. Log saved to "%LOG_FILE%".
) else (
    echo [ERROR] Log file "%LOG_FILE%" was not created. Something might have gone wrong.
    exit /b 1
)

pause