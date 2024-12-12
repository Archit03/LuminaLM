#!/bin/bash

# Ensure the script exits on errors and undefined variables
set -eu

# Define variables
TOKENIZER_SCRIPT="tokenizer.py"
VALIDATION_SCRIPT="token_validation.py"
LOCAL_DATA_DIR="/content/LuminaLM/Data"
TOKENIZER_OUTPUT_PATH="./lumina_tokenizer.json"
VALIDATION_OUTPUT="./validation_results.txt"
LOG_FILE="lumina_tokenization.log"

# Print script configuration
echo "=== LuminaLM Tokenizer Configuration ==="
echo "Data Directory: $LOCAL_DATA_DIR"
echo "Output Path: $TOKENIZER_OUTPUT_PATH"
echo "Log File: $LOG_FILE"
echo "=================================="

# Check if the tokenizer script exists
if [[ ! -f $TOKENIZER_SCRIPT ]]; then
  echo "Error: $TOKENIZER_SCRIPT not found!"
  exit 1
fi

# Check if the validation script exists
if [[ ! -f $VALIDATION_SCRIPT ]]; then
  echo "Error: $VALIDATION_SCRIPT not found!"
  exit 1
fi

# Run the tokenizer script with error handling
echo "Running tokenizer script..."
if ! python3 $TOKENIZER_SCRIPT \
    --local_data_path="$LOCAL_DATA_DIR" \
    --vocab_size=32000 \
    --min_frequency=3 \
    --limit_alphabet=1000 \
    --model_prefix="lumina" \
    --log_file="$LOG_FILE"; then
    echo "Error: Tokenizer script failed!"
    exit 1
fi

# Check if the tokenizer output was created
if [[ ! -f $TOKENIZER_OUTPUT_PATH ]]; then
  echo "Error: Tokenizer output file $TOKENIZER_OUTPUT_PATH not created!"
  exit 1
fi

# Run the validation script with error handling
echo "Running validation script..."
if ! python3 $VALIDATION_SCRIPT --tokenizer_path "$TOKENIZER_OUTPUT_PATH" > "$VALIDATION_OUTPUT"; then
    echo "Error: Validation script failed!"
    exit 1
fi

# Check if the validation output was created
if [[ ! -f $VALIDATION_OUTPUT ]]; then
  echo "Error: Validation output file $VALIDATION_OUTPUT not created!"
  exit 1
fi

# Display final statistics
echo -e "\n=== LuminaLM Tokenizer Status ==="
if [[ -f "$LOG_FILE" ]]; then
    echo "Tokenization log available at: $LOG_FILE"
fi
echo "Tokenizer saved at: $TOKENIZER_OUTPUT_PATH"
echo "Validation results saved at: $VALIDATION_OUTPUT"
