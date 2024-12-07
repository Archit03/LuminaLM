import logging
import os
import json
import mimetypes
import hashlib  # Added for sanitizing filenames
import multiprocessing  # Added for parallel processing
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from concurrent.futures import ProcessPoolExecutor, as_completed  # For parallel dataset processing
from itertools import islice  # For efficient chunking of file lines or JSON objects
from tqdm import tqdm  # For progress bars

import torch
from torch import Tensor
from datasets import load_dataset  # For loading HuggingFace datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
# Set up logging
LOG_FILE = "medical_tokenization.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

class TokenizationUtilities:
    """
    Tokenization utilities for complex NLP preprocessing.
    """

    @staticmethod
    def dynamic_padding(
        input_ids: List[Tensor],
        attention_masks: List[Tensor],
        padding_value: int = 0,
        padding_side: str = 'right',
        max_length: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Dynamically pad input sequences to a consistent length.

        :param input_ids: List of input token IDs
        :param attention_masks: List of attention masks
        :param padding_value: Value to use for padding
        :param padding_side: Side to pad ('left' or 'right')
        :param max_length: Maximum sequence length
        :return: Padded input_ids and attention_masks
        """
        # Get device from first tensor or default to CUDA if available
        device = input_ids[0].device if input_ids else (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Determine max length if not specified
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)

        # Pad input_ids
        padded_input_ids = []
        for ids in input_ids:
            pad_length = max_length - len(ids)
            if padding_side == 'right':
                padded_ids = torch.cat([
                    ids,
                    torch.full((pad_length,), padding_value, dtype=ids.dtype)
                ])
            else:  # left padding
                padded_ids = torch.cat([
                    torch.full((pad_length,), padding_value, dtype=ids.dtype),
                    ids
                ])
            padded_input_ids.append(padded_ids)

        # Pad attention masks
        padded_attention_masks = []
        for mask in attention_masks:
            pad_length = max_length - len(mask)
            if padding_side == 'right':
                padded_mask = torch.cat([
                    mask,
                    torch.zeros(pad_length, dtype=mask.dtype)
                ])
            else:  # left padding
                padded_mask = torch.cat([
                    torch.zeros(pad_length, dtype=mask.dtype),
                    mask
                ])
            padded_attention_masks.append(padded_mask)

        # Move final tensors to appropriate device
        return (torch.stack(padded_input_ids).to(device), 
                torch.stack(padded_attention_masks).to(device))

    @staticmethod
    def create_segment_ids(
        input_ids: Tensor,
        separator_token_id: int,
        cls_token_id: int
    ) -> Tensor:
        """
        Create segment IDs for multi-segment inputs (e.g., sentence pairs).

        :param input_ids: Padded input token IDs
        :param separator_token_id: ID of the separator token
        :param cls_token_id: ID of the classification token
        :return: Tensor of segment IDs
        """
        # Create segment IDs on same device as input
        device = input_ids.device
        segment_ids = torch.zeros_like(input_ids).to(device)
        for i, seq in enumerate(input_ids):
            # Find separator and CLS token positions
            sep_positions = (seq == separator_token_id).nonzero(as_tuple=True)[0]
            cls_position = (seq == cls_token_id).nonzero(as_tuple=True)[0]

            # Assign segment IDs
            if len(sep_positions) > 0:
                segment_ids[i, cls_position[0]:sep_positions[0] + 1] = 0
                segment_ids[i, sep_positions[0] + 1:] = 1

        return segment_ids

    @staticmethod
    def generate_masked_lm_inputs(
        input_ids: Tensor,
        mask_probability: float = 0.15,
        mask_token_id: int = 103,  # Typically the [MASK] token
        special_token_ids: Optional[List[int]] = None,
        vocab_size: int = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate masked language model inputs for pretraining.

        :param input_ids: Original input token IDs (batch_size x seq_length)
        :param mask_probability: Probability of masking a token
        :param mask_token_id: ID of the mask token
        :param special_token_ids: List of token IDs to never mask
        :param vocab_size: Vocabulary size for random token replacement
        :return: Masked input_ids, masked labels, and masking indicator
        """
        device = input_ids.device
        if special_token_ids is None:
            special_token_ids = []

        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        batch_size, seq_length = input_ids.shape

        # Create mask for tokens that can be masked
        maskable_positions = torch.ones_like(input_ids, dtype=torch.bool)
        for special_id in special_token_ids:
            maskable_positions &= (input_ids != special_id)

        # Randomly decide which tokens to mask
        probabilities = torch.full(input_ids.shape, mask_probability).to(device)
        mask_probabilities = torch.bernoulli(probabilities).bool() & maskable_positions

        # Apply masking
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & mask_probabilities
        masked_input_ids[indices_replaced] = mask_token_id

        # Apply random token replacement
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & mask_probabilities & ~indices_replaced
        if vocab_size is None:
            vocab_size = int(input_ids.max()) + 1
        random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=input_ids.dtype)
        masked_input_ids[indices_random] = random_tokens[indices_random]

        # The rest 10% keep the original tokens
        # For tokens not masked, set labels to -100 so they are ignored in loss calculation
        labels[~mask_probabilities] = -100

        return masked_input_ids, labels, mask_probabilities

class HybridTokenizationStrategy:
    """Tokenization strategy supporting both autoregressive and bidirectional processing."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.utilities = TokenizationUtilities()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def autoregressive_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True
    ):
        """Encode text for left-to-right prediction tasks with causal masking."""
        encodings = self.tokenizer.encode_batch(texts)
        input_ids, attention_masks = [], []

        for encoding in encodings:
            ids = encoding.ids[:max_length] if truncation else encoding.ids
            input_ids.append(torch.tensor(ids, device=self.device))
            seq_len = len(ids)
            mask = torch.ones(seq_len, dtype=torch.long, device=self.device)
            attention_masks.append(mask)

        # Apply dynamic padding if required
        if padding:
            input_ids, attention_masks = self.utilities.dynamic_padding(
                input_ids,
                attention_masks
            )

        # Create causal masks
        causal_masks = []
        for seq_len in attention_masks.sum(dim=1).tolist():
            causal_mask = torch.tril(torch.ones(seq_len, seq_len))
            pad_size = input_ids.shape[1] - seq_len
            causal_mask = torch.nn.functional.pad(causal_mask, (0, pad_size, 0, pad_size), value=0)
            causal_masks.append(causal_mask)

        causal_masks = torch.stack(causal_masks)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'causal_mask': causal_masks
        }

    def bidirectional_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True,
        add_special_tokens: bool = True
    ):
        """Encode text with full contextual understanding for bidirectional tasks."""
        encodings = self.tokenizer.encode_batch(
            texts,
            add_special_tokens=add_special_tokens
        )
        input_ids, attention_masks = [], []

        for encoding in encodings:
            ids = encoding.ids[:max_length] if truncation else encoding.ids
            input_ids.append(torch.tensor(ids, device=self.device))
            mask = torch.ones(len(ids), dtype=torch.long, device=self.device)
            attention_masks.append(mask)

        # Apply dynamic padding if required
        if padding:
            input_ids, attention_masks = self.utilities.dynamic_padding(
                input_ids,
                attention_masks
            )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

    def hybrid_encode(
        self,
        texts: List[str],
        task_type: str = 'auto',
        **kwargs
    ):
        """Dynamically choose encoding strategy based on task type."""
        if task_type == 'auto':
            return self.autoregressive_encode(texts, **kwargs)
        elif task_type == 'bi':
            return self.bidirectional_encode(texts, **kwargs)
        else:
            raise ValueError(f"Invalid task type: {task_type}. Use 'auto' or 'bi'.")

class MedicalTokenizer:
    """Tokenizer class that combines both autoregressive and bidirectional encoding strategies."""

    def __init__(self, vocab_size: int = 60000, min_frequency: int = 2):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens
        )
        self.strategy = HybridTokenizationStrategy(self.tokenizer)

    def train(self, files: List[str], save_path: str):
        """
        Train the tokenizer on the given files and save it to the specified path.

        :param files: List of file paths to use for training
        :param save_path: Path to save the trained tokenizer
        """
        logging.info("Starting tokenizer training...")
        self.tokenizer.train(files, self.trainer)
        self.tokenizer.save(save_path)
        logging.info(f"Tokenizer saved to {save_path}")

    def encode(self, texts: List[str], task_type: str = 'auto', **kwargs):
        """
        Encode a list of texts using the specified task type ('auto' or 'bi').

        :param texts: List of texts to encode
        :param task_type: 'auto' for autoregressive or 'bi' for bidirectional encoding
        :return: Encoded inputs
        """
        return self.strategy.hybrid_encode(texts, task_type=task_type, **kwargs)

class FileValidator:
    """Validates and sanitizes file uploads."""
    
    ALLOWED_EXTENSIONS: Set[str] = {'.txt', '.csv', '.json', '.jsonl', '.text'}
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_MIMETYPES: Set[str] = {
        'text/plain', 'text/csv', 'application/json',
        'application/x-json', 'text/json'
    }

    @classmethod
    def validate_file(cls, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate a file for processing.
        
        :param file_path: Path to the file
        :return: Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        try:
            if not file_path.exists():
                return False, "File does not exist"

            if not file_path.is_file():
                return False, "Path is not a file"

            if file_path.suffix.lower() not in cls.ALLOWED_EXTENSIONS:
                return False, f"Unsupported file extension. Allowed: {cls.ALLOWED_EXTENSIONS}"

            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type not in cls.ALLOWED_MIMETYPES:
                return False, f"Invalid file type: {mime_type}"

            if file_path.stat().st_size > cls.MAX_FILE_SIZE:
                return False, f"File too large. Maximum size: {cls.MAX_FILE_SIZE/1024/1024}MB"

            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename for safe storage.
        
        :param filename: Original filename
        :return: Sanitized filename
        """
        # Create a hash of the original filename
        filename_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        # Get the extension
        ext = Path(filename).suffix
        # Create a safe filename
        safe_name = f"file_{filename_hash}{ext}"
        return safe_name

class DatasetProcessor:
    """Enhanced dataset processor with parallel processing and better error handling."""

    def __init__(self, datasets: List[Dict[str, Any]], local_data_path: str):
        self.datasets = datasets
        self.local_data_path = Path(local_data_path)
        self.processed_files = []
        self.file_validator = FileValidator()
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)

    def process_chunk(self, chunk: List[Any]) -> List[str]:
        """Process a chunk of data in parallel."""
        processed_texts = []
        for item in chunk:
            text = self.extract_text(item)
            if text.strip():
                processed_texts.append(text)
        return processed_texts

    def parallel_process_dataset(self, dataset, chunk_size: int = 1000) -> List[str]:
        """
        Process dataset in parallel chunks.
        
        :param dataset: Dataset to process
        :param chunk_size: Size of chunks for parallel processing
        :return: List of processed texts
        """
        chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]
        processed_texts = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
            
            with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                for future in as_completed(futures):
                    try:
                        processed_texts.extend(future.result())
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Error processing chunk: {str(e)}")

        return processed_texts

    def process_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Enhanced file processing with validation and error handling."""
        file_path = Path(file_path)
        
        # Validate file
        is_valid, error_msg = self.file_validator.validate_file(file_path)
        if not is_valid:
            logging.error(f"File validation failed: {error_msg}")
            return None

        # Create safe output filename
        safe_filename = self.file_validator.sanitize_filename(file_path.name)
        output_file = self.local_data_path / f"processed_{safe_filename}"

        try:
            ext = file_path.suffix.lower()
            
            if ext == '.csv':
                return self._process_csv_file(file_path, output_file)
            elif ext in {'.json', '.jsonl'}:
                return self._process_json_file(file_path, output_file)
            elif ext in {'.txt', '.text'}:
                return self._process_text_file(file_path, output_file)
            else:
                logging.warning(f"Unsupported file format: {ext}")
                return None

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            if output_file.exists():
                output_file.unlink()  # Clean up partial file
            return None

    def _process_csv_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """Process CSV files with chunking for large files."""
        try:
            import pandas as pd
            chunk_size = 10000  # Adjust based on available memory
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                    processed_texts = self.parallel_process_dataset(
                        chunk.to_dict('records'),
                        chunk_size=1000
                    )
                    for text in processed_texts:
                        f.write(f"{text}\n")
            
            return str(output_path)
        except Exception as e:
            logging.error(f"Error processing CSV file: {str(e)}")
            return None

    def _process_json_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """Process JSON files with streaming for large files."""
        try:
            import json
            from itertools import islice

            def json_stream(file_obj):
                """Stream JSON objects for memory efficiency."""
                buffer = []
                for line in file_obj:
                    buffer.append(line)
                    if line.strip().endswith('}') or line.strip().endswith(']'):
                        try:
                            yield json.loads(''.join(buffer))
                            buffer = []
                        except json.JSONDecodeError:
                            continue

            with open(input_path, 'r', encoding='utf-8') as f_in:
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    # Process in chunks of 1000 objects
                    while True:
                        chunk = list(islice(json_stream(f_in), 1000))
                        if not chunk:
                            break
                        processed_texts = self.parallel_process_dataset(chunk)
                        for text in processed_texts:
                            f_out.write(f"{text}\n")

            return str(output_path)
        except Exception as e:
            logging.error(f"Error processing JSON file: {str(e)}")
            return None

    def _process_text_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """Process text files with buffered reading."""
        try:
            chunk_size = 10000  # lines per chunk
            with open(input_path, 'r', encoding='utf-8') as f_in:
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    while True:
                        lines = list(islice(f_in, chunk_size))
                        if not lines:
                            break
                        processed_texts = self.parallel_process_dataset(lines)
                        for text in processed_texts:
                            f_out.write(f"{text}\n")

            return str(output_path)
        except Exception as e:
            logging.error(f"Error processing text file: {str(e)}")
            return None

def main():
    """Main execution for medical tokenizer training."""
    datasets = [
        {"name": "openwebtext", "trust_remote_code": True},
        {"name": "pubmed_qa", "configs": ["pqa_artificial", "pqa_labeled", "pqa_unlabeled"]},
    ]
    local_data_path = "Update this"

    dataset_processor = DatasetProcessor(datasets, local_data_path)
    tokenizer = MedicalTokenizer(vocab_size=60000, min_frequency=3)

    try:
        processed_data = dataset_processor.process()
        tokenizer.train(processed_data, "Medical_tokenizer.json")

        logging.info("Tokenizer training completed successfully.")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
