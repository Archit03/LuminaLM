@echo off
setlocal EnableDelayedExpansion

:: Install required packages
pip install transformers torch torchvision torchaudio datasets tokenizers pandas pillow tqdm

:: Define variables
set TOKENIZER_SCRIPT=tokenizer.py
set LOCAL_DATA_DIR=C:\Users\ASUS\Desktop\LuminaLM\Data
set VOCAB_SIZE=60000
set MIN_FREQUENCY=2
set LOG_FILE=medical_tokenization.log

:: Run the tokenizer script
echo Running tokenizer script...
python %TOKENIZER_SCRIPT% --local_data_path "%LOCAL_DATA_DIR%" --vocab_size %VOCAB_SIZE% --min_frequency %MIN_FREQUENCY% --log_file %LOG_FILE%

:: Check if the log file was created
if exist %LOG_FILE% (
    echo Tokenizer process completed. Check %LOG_FILE% for details.
) else (
    echo Error: Log file not created!
    exit /b 1
)

pause