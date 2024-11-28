#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
CONFIG_PATH="config.yaml"
LOCAL_DATA_DIR="LuminaLM/Data"  # Set your local data directory here
TOKENIZER_PATH="./medical_tokenizer.json"
CHECKPOINT=""  # Set this to the path of a checkpoint if resuming training

# Parse command-line arguments for optional checkpoint
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="--checkpoint $2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Run the training script
python embeddings.py \
    --config "$CONFIG_PATH" \
    --local_data_dir "$LOCAL_DATA_DIR" \
    $CHECKPOINT
