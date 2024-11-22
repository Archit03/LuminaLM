#!/bin/bash

# Variables
CONFIG_PATH="config.yaml"
TOKENIZER_PATH="LuminaLMtokenizer.json"
LOCAL_DATA_DIR="/mnt/c/Users/ASUS/Desktop/LuminaLM/Data"  # Adjusted for Linux path

# Run tokenizer
python tokenizer.py --config $CONFIG_PATH --tokenizer_path $TOKENIZER_PATH --local_data_dir $LOCAL_DATA_DIR

# Run embeddings
python embeddings.py --config $CONFIG_PATH --tokenizer_path $TOKENIZER_PATH --local_data_dir $LOCAL_DATA_DIR
