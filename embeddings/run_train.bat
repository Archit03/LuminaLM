@echo off
REM Windows Batch Script to Train LuminaLM Embeddings Model

REM Variables
SET CONFIG_PATH=config.yaml
SET LOCAL_DATA_DIR=C:\Users\ASUS\Desktop\LuminaLM\Data
SET TOKENIZER_PATH=C:\Users\ASUS\Desktop\LuminaLM\embeddings\Medical_tokenizer.json

REM Run Embeddings Training Script without a Checkpoint
python embeddings.py ^
    --config "%CONFIG_PATH%" ^
    --local_data_dir "%LOCAL_DATA_DIR%"

REM Pause to see any error messages
pause
