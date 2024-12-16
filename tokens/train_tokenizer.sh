#!/bin/bash

# Enable error handling
set -e
set -o pipefail

# Set environment variables
PYTHON_PATH="python"
SCRIPT_PATH="tokenizer.py"
LOG_FILE="tokenizer.log"
MIN_FREQ=2
CHUNK_SIZE=10000

# Check Python installation
if ! command -v $PYTHON_PATH &> /dev/null; then
    echo "Error: Python not found. Please ensure Python is installed and in PATH."
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Tokenizer script not found at $SCRIPT_PATH"
    exit 1
fi

# Check CUDA availability and echo status
echo "Checking hardware configuration..."
$PYTHON_PATH -c "import torch; print(f'I\'m using {\"CUDA\" if torch.cuda.is_available() else \"CPU\"} to do my job')"
echo

# Run the tokenizer with enhanced configuration
echo "Starting tokenizer training..."
$PYTHON_PATH "$SCRIPT_PATH" \
    --min_freq $MIN_FREQ \
    --log "$LOG_FILE" \
    --chunk_size $CHUNK_SIZE \
    --workers 8

if [ $? -ne 0 ]; then
    echo "Error: Tokenizer training failed. Check $LOG_FILE for details."
    exit 1
else
    echo "Tokenizer training completed successfully."
fi
