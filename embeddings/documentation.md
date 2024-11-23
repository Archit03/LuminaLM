# LuminaLM: Transformer-Based Embeddings Model

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
   - [Training the Tokenizer](#training-the-tokenizer)
   - [Training the Model](#training-the-model)
6. [Scripts Overview](#scripts-overview)
   - [`tokenizer.py`](#tokenizerpy)
   - [`train_transformer_embeddings.py`](#train_transformer_embeddingspy)
   - [`run_train.sh`](#run_trainsh)
7. [Components Explanation](#components-explanation)
   - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
   - [Model Architecture](#model-architecture)
   - [Training Loop](#training-loop)
   - [Embedding Generation](#embedding-generation)
   - [Visualization](#visualization)
8. [Pinecone Integration](#pinecone-integration)
9. [Environment Variables](#environment-variables)
10. [Troubleshooting](#troubleshooting)
11. [Extending the Project](#extending-the-project)
12. [Acknowledgements](#acknowledgements)

---

## Introduction

LuminaLM is a transformer-based embeddings model designed to generate high-quality embeddings from text data. It leverages transformer architectures to learn rich representations of textual data, which can be used in various downstream tasks such as semantic search, clustering, and classification.

The project includes scripts for:

- Training a custom tokenizer.
- Training the transformer model.
- Generating embeddings and saving them to PineconeDB.
- Visualizing training metrics and embeddings.

---

## Features

- **Custom Tokenizer Training**: Ability to train a tokenizer on your specific dataset.
- **Transformer Model Training**: Train a transformer model with customizable hyperparameters.
- **Multiprocessing for Tokenization**: Speeds up data preprocessing using multiprocessing.
- **Learning Rate Scheduling**: Implements a cosine annealing learning rate scheduler.
- **Validation at Regular Intervals**: Allows for intermediate validation during training.
- **Gradient Noise Addition**: Optionally adds noise to gradients for better generalization.
- **Dataset Caching**: Efficient data handling to prevent redundant computations.
- **Distributed Training**: Supports distributed training using PyTorch's distributed data parallel (DDP).
- **Embedding Generation**: Generates embeddings and saves them to PineconeDB.
- **Visualization Tools**: Visualizes training metrics and embeddings using PCA and t-SNE.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Tokenizers
- Hugging Face Datasets
- Pinecone Client
- Additional Python packages:
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `tqdm`
  - `yaml`
  - `torchvision`
  - `python-dotenv` (optional, for environment variable management)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/luminalm.git
   cd luminalm
   ```

2. **Create a Virtual Environment (Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install packages manually:*

   ```bash
   pip install torch tokenizers datasets pinecone-client tqdm matplotlib scikit-learn seaborn pyyaml
   ```

---

## Configuration

All configurations are managed through the `config.yaml` file. This file contains settings for the model, tokenizer, data, training parameters, logging, checkpointing, visualization, distributed training, and Pinecone integration.

### Sample `config.yaml`

```yaml
model:
  d_model: 512
  src_seq_len: 512
  batch_size: 32
  learning_rate: 5e-5
  epochs: 5
  patience: 3

tokenizer:
  save_path: "embeddings/tokenizer.json"
  load_path: "embeddings/tokenizer.json"
  type: 'WordPiece'
  vocab_size: 30522
  max_samples: 100000
  datasets:
    - name: 'openwebtext'
      split: 'train'
      text_column: 'text'
    - name: 'pubmed_qa'
      split: 'train'
      text_column: 'text'

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  max_samples: 100000

training:
  use_mixed_precision: true
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  warmup_steps: 1000
  weight_decay: 0.01
  dropout: 0.1
  validation_steps: 500
  gradient_noise_std: 0.0

logging:
  level: INFO
  save_dir: "logs"
  metrics_file: "metrics.json"

checkpointing:
  save_dir: "checkpoints"
  save_frequency: 1
  keep_best_n: 3

visualization:
  plot_dir: "plots"
  sample_size: 1000
  embedding_dims: 3

distributed:
  backend: "nccl"
  world_size: -1  # auto-detect
  init_method: "env://"

pinecone:
  api_key: "${PINECONE_API_KEY}"  # Use environment variable
  environment: "${PINECONE_ENVIRONMENT}"  # Use environment variable
  index_name: "luminalm-embeddings"
```

**Note**: Replace placeholders with actual values or set them as environment variables.

---

## Usage

### Training the Tokenizer

1. **Set Environment Variables**

   ```bash
   export PINECONE_API_KEY="your_pinecone_api_key"
   export PINECONE_ENVIRONMENT="your_pinecone_environment"
   ```

2. **Run the Tokenizer Script**

   ```bash
   python tokenizer.py --config config.yaml --tokenizer_path embeddings/tokenizer.json --local_data_dir /path/to/your/data
   ```

   - `--config`: Path to the configuration file.
   - `--tokenizer_path`: Path where the tokenizer will be saved.
   - `--local_data_dir`: Path to your local data directory containing `.txt` files.

### Training the Model

Use the provided shell script or run the training script directly.

#### Using `run_train.sh`

1. **Make the Script Executable**

   ```bash
   chmod +x run_train.sh
   ```

2. **Run the Script**

   ```bash
   ./run_train.sh --local_data_dir /path/to/your/data
   ```

   - To resume from a checkpoint:

     ```bash
     ./run_train.sh --local_data_dir /path/to/your/data --checkpoint checkpoints/best_model.pt
     ```

#### Running the Training Script Directly

```bash
python train_transformer_embeddings.py --config config.yaml --local_data_dir /path/to/your/data --tokenizer_path embeddings/tokenizer.json
```

---

## Scripts Overview

### `tokenizer.py`

- **Purpose**: Trains and saves a tokenizer based on your data.
- **Usage**:

  ```bash
  python tokenizer.py --config config.yaml --tokenizer_path embeddings/tokenizer.json --local_data_dir /path/to/your/data
  ```

- **Key Functions**:
  - Loads datasets specified in the configuration.
  - Trains a tokenizer (`WordPiece` or `BPE`).
  - Saves the tokenizer to the specified path.

### `train_transformer_embeddings.py`

- **Purpose**: Trains the transformer model, generates embeddings, and saves them to PineconeDB.
- **Usage**:

  ```bash
  python train_transformer_embeddings.py --config config.yaml --local_data_dir /path/to/your/data --tokenizer_path embeddings/tokenizer.json
  ```

- **Key Functions**:
  - Loads data and prepares datasets.
  - Initializes the model and optimizer.
  - Implements the training loop with validation.
  - Generates embeddings from test data.
  - Saves embeddings to PineconeDB.
  - Creates visualizations of metrics and embeddings.

### `run_train.sh`

- **Purpose**: Shell script to automate running the tokenizer and training scripts.
- **Usage**:

  ```bash
  ./run_train.sh --local_data_dir /LuminaLM/Data
  ```

- **Parameters**:
  - `--local_data_dir`: Path to your local data directory.
  - `--checkpoint`: (Optional) Path to a checkpoint file to resume training.

---

## Components Explanation

### Data Loading and Preprocessing

- **Local Data Loading**: Reads `.txt` files from the specified directory.
- **Dataset Loading**: Uses Hugging Face Datasets to load datasets like OpenWebText and medical datasets.
- **Tokenization**: Uses the trained tokenizer to convert text into token IDs.
- **Multiprocessing**: Tokenization is parallelized using Python's `multiprocessing` module to speed up processing.

### Model Architecture

- **Transformer Model**: Built using PyTorch, with customizable parameters specified in `config.yaml`.
- **Sequence Lengths**: Supports configurable source and target sequence lengths.
- **Embedding Dimension**: Configurable via `d_model` parameter.

### Training Loop

- **Optimizer**: Uses `AdamW` optimizer with weight decay.
- **Learning Rate Scheduler**: Implements a cosine annealing scheduler (`CosineAnnealingLR`).
- **Mixed Precision**: Optionally uses mixed precision training for performance improvements.
- **Gradient Accumulation**: Supports accumulating gradients over multiple batches.
- **Validation**: Performs validation at the end of each epoch and optionally at regular steps.
- **Early Stopping**: Stops training if validation loss does not improve for a specified number of epochs.

### Embedding Generation

- **Encoding**: Uses the trained model to generate embeddings from input IDs.
- **Saving to PineconeDB**: Embeddings are saved to PineconeDB in chunks to manage memory usage.
- **Indexing**: Each embedding is associated with a unique ID for retrieval.

### Visualization

- **Metrics Plotting**: Plots training and validation loss, accuracy, and perplexity over epochs.
- **Embedding Visualization**: Uses PCA and t-SNE to reduce embedding dimensions for visualization.
- **Plots**: Saved in the directory specified in the configuration (`visualization.plot_dir`).

---

## Pinecone Integration

- **Purpose**: Stores generated embeddings for efficient similarity search and retrieval.
- **Setup**:
  - **API Key and Environment**: Set `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` as environment variables.
  - **Index Creation**: The script checks if the specified index exists; if not, it creates one.
- **Usage**: Embeddings are upserted into the Pinecone index in batches.

---

## Environment Variables

- **Security**: API keys and sensitive information are managed via environment variables to prevent exposure.
- **Variables Used**:
  - `PINECONE_API_KEY`: Your Pinecone API key.
  - `PINECONE_ENVIRONMENT`: The Pinecone environment (e.g., `us-east-1`).
- **Setting Variables**:

  ```bash
  export PINECONE_API_KEY="your_pinecone_api_key"
  export PINECONE_ENVIRONMENT="your_pinecone_environment"
  ```

- **Optional**: Use a `.env` file and `python-dotenv` to manage environment variables.

---

## Troubleshooting

- **ModuleNotFoundError**: Ensure all required packages are installed and available in your environment.
- **Memory Errors**: Reduce batch sizes or sequence lengths if you encounter memory issues.
- **API Key Errors**: Verify that `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` are correctly set.
- **Data Loading Issues**: Check that data directories and dataset names are correct and accessible.
- **Checkpoint Loading Errors**: Ensure the checkpoint path is correct and the file is not corrupted.
- **Visualization Errors**: Confirm that `matplotlib` and `seaborn` are installed and functioning.

---

## Extending the Project

- **Adding New Datasets**: Update the `tokenizer.datasets` and data loading functions to include new datasets.
- **Modifying the Model**: Adjust the model architecture in the `model` module as needed.
- **Custom Tokenizers**: Experiment with different tokenizer types or configurations.
- **Advanced Training Techniques**: Implement techniques like learning rate warm-up, different schedulers, or regularization methods.
- **Distributed Training**: Expand the distributed training capabilities for multi-node setups.
- **Integration with Other Databases**: Modify the embedding saving functions to work with different vector databases.

---

## Acknowledgements

- **Hugging Face**: For the `transformers` and `datasets` libraries.
- **PyTorch**: For providing a powerful deep learning framework.
- **Pinecone**: For enabling efficient vector storage and retrieval.
- **OpenAI**: For inspiring advancements in language models.

---
