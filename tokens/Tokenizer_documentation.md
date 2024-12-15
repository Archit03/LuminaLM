 # Medical Tokenizer Script Documentation

## Overview
This script facilitates the training, validation, and use of a tokenizer optimized for medical text data. Designed for high-performance preprocessing, it supports dynamic padding, attention masking, truncation, and encoding utilities tailored for domain-specific text.

---

## Features
- **Dynamic Padding**: Automatically adjusts padding length to accommodate varying input sequence sizes.
- **Attention Mask Generation**: Creates masks to distinguish between valid tokens and padding for efficient model attention handling.
- **Truncation Support**: Ensures input sequences do not exceed the specified maximum length.
- **Flexible Encoding**: Converts raw text into input IDs and attention masks with support for dynamic tensor return.
- **Hybrid Tokenization**: Supports autoregressive and bidirectional encoding schemes.
- **Domain-Specific Preprocessing**: Handles medical abbreviations, anonymization of Protected Health Information (PHI), and standardization of units.
- **Dataset Support**: Processes both local files and Hugging Face datasets efficiently.
- **Logging and Visualization**: Includes robust logging and graph generation for validation and performance analysis.

---

## Configuration Parameters
The `Config` class allows flexible customization, including:
- **`vocab_size`**: Size of the tokenizer’s vocabulary.
- **`min_frequency`**: Minimum frequency for Byte Pair Encoding (BPE) merges.
- **`local_data_path`**: Directory to store processed data and tokenizers.
- **`allowed_extensions`**: Supported file extensions for local dataset processing.

---

## Core Components

### **1. MedicalTokenizer**
This is the primary class used for training and utilizing the tokenizer.

#### Key Methods:
- **`train(files: List[str], save_path: str)`**:
  - Trains the tokenizer using the provided text files.
  - Saves the trained tokenizer as `Medical_tokenizer.json`.

- **`encode(texts: List[str], padding: bool, truncation: bool, max_length: int, return_tensors: str)`**:
  - Encodes input text into token IDs and attention masks.
  - Supports options for padding, truncation, and returning tensors (`"pt"` for PyTorch, `"np"` for NumPy).

- **`_setup_normalizer()`**:
  - Applies preprocessing rules specific to medical text, including removing irrelevant characters and normalizing units.

---

### **2. DatasetProcessor**
Processes datasets for tokenizer training and evaluation.

#### Key Methods:
- **`_process_local_dataset(config: Config)`**:
  - Reads local text files and preprocesses them for training.

- **`_process_huggingface_dataset(config: Config)`**:
  - Downloads and processes datasets directly from Hugging Face.

---

### **3. TokenizationUtilities**
Utility class for tokenization-related operations.

#### Key Methods:
- **`dynamic_padding(input_ids: List[List[int]], attention_masks: List[List[int]], padding_value: int, max_length: int)`**:
  - Dynamically pads sequences to the specified `max_length`.

- **`create_attention_mask(input_ids: List[List[int]], padding_token_id: int)`**:
  - Creates attention masks to identify valid tokens in sequences.

- **`generate_masked_lm_inputs(input_ids: List[List[int]], mask_probability: float)`**:
  - Prepares inputs for Masked Language Modeling (MLM) tasks.

---

## Validation Workflow
The validation script ensures the tokenizer is working as intended. Key tasks include:
1. **Dynamic Padding and Truncation**:
   - Automatically adjusts input sequences to `max_length`.
   - Logs truncated tokens for debugging.

2. **Attention Mask Creation**:
   - Validates the creation of attention masks for all input sequences.

3. **Visualization**:
   - Generates graphs to analyze performance:
     - **Sequence Lengths**: Displays original vs. truncated lengths.
     - **Attention Mask Coverage**: Shows the proportion of valid tokens per sequence.
     - **Padding Efficiency**: Highlights the ratio of used tokens to total padding.

---

## Sample Workflow
### **Training the Tokenizer**
```bash
python tokenizer.py --config dataset_config.yaml --vocab_size 60000 --min_freq 2
```

### **Validation**
```bash
python validation.py
```

---

## Outputs
### **Files**
- **Tokenizer Model**: `Medical_tokenizer.json` (saved in the `tokens/` directory).
- **Validation Logs**: `tokens/validation.log`.
- **Visualization Outputs**:
  - `sequence_lengths.png`
  - `attention_mask_coverage.png`
  - `padding_efficiency.png`

---

## Logging
Logs are saved to `tokens/validation.log` and include:
- Tokenizer loading status.
- Details of truncated tokens and sequence lengths.
- Shapes of input IDs and attention masks.

---

## Requirements
- **Python Version**: 3.8 or higher
- **Dependencies**:
  - `torch`: PyTorch for tensor handling.
  - `tokenizers`: For tokenizer training and processing.
  - `transformers`: Hugging Face library for NLP models.
  - `datasets`: Dataset handling utilities.
  - `matplotlib`: Visualization library.
  - `tqdm`, `psutil`: Progress bars and system utilities.

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Limitations and Recommendations
### **Limitations**:
- Requires substantial memory for large datasets.
- Truncation may lead to loss of important context in very long medical texts.

### **Recommendations**:
1. **Dynamic Batching**:
   - Implement dynamic batching for optimal memory usage.
2. **Dataset Expansion**:
   - Test on larger datasets (e.g., full clinical reports or research papers) for scalability.
3. **Truncation Analysis**:
   - Evaluate the impact of truncated tokens on downstream tasks like summarization or question answering.

