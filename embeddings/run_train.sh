#!/bin/bash

# Variables
CONFIG_PATH="config.yaml"
TOKENIZER_PATH="LuminaLMtokenizer.json"
LOCAL_DATA_DIR="/mnt/c/Users/ASUS/Desktop/LuminaLM/Data"  # Adjust this path as needed
CHECKPOINT=""  # Set this to the path of a checkpoint if resuming training

# Parse command-line arguments for optional checkpoint
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="--checkpoint $2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run tokenizer
python tokenizer.py --config "$CONFIG_PATH" --tokenizer_path "$TOKENIZER_PATH" --local_data_dir "$LOCAL_DATA_DIR"

# Run the training script
python train_transformer_embeddings.py --config "$CONFIG_PATH" --local_data_dir "$LOCAL_DATA_DIR" --tokenizer_path "$TOKENIZER_PATH" $CHECKPOINT
