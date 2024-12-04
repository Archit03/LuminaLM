#!/bin/bash

# This script will execute the training using the specified configuration file.

# Path to the Python interpreter, adjust if needed
PYTHON=python3

# Check if the Python interpreter exists
if ! command -v $PYTHON &> /dev/null; then
    echo "Error: Python interpreter not found. Please install Python 3 or adjust the path."
    exit 1
fi

# Path to the config file
CONFIG_PATH="config.yaml"

# Check if the configuration file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at '$CONFIG_PATH'. Please create or provide a valid config file."
    exit 1
fi

# Running the training script with the YAML config file
$PYTHON train.py --config_path "$CONFIG_PATH"

# Check if the training script ran successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training script encountered an error."
    exit 1
fi
