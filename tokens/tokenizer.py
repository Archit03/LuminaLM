import argparse
import logging
import os
import json
import mimetypes
import hashlib
import multiprocessing
import traceback
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Generator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from itertools import islice
from tqdm import tqdm
import torch
from torch import Tensor
from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
import pandas as pd
from PIL import Image
import io
from transformers import ViTImageProcessor, CLIPProcessor
import re
import psutil
import shutil
import tempfile
import time
from contextlib import contextmanager
from functools import partial
import gc
import unittest
from utils import with_retry, managed_temp_file, split_chunk
import asyncio
import random
import yaml

###############################################################################
# Configuration
###############################################################################
class Config:
    """
    Configuration class to hold parameters for dataset processing and tokenizer training.
    """
    def __init__(
        self, 
        local_data_path: str,
        vocab_size: int = 60000,
        min_frequency: int = 2,
        max_file_size_mb: int = 100,
        allowed_extensions: Set[str] = {'.txt', '.csv', '.json', '.jsonl', '.text', 
                                      '.jpg', '.jpeg', '.png', '.webp'},
        allowed_mimetypes: Set[str] = {
            'text/plain', 'text/csv', 'application/json',
            'application/x-json', 'text/json',
            'image/jpeg', 'image/png', 'image/webp'
        },
        image_processor: str = 'google/vit-base-patch16-224',
        chunk_size: int = 1000000,
        processing_workers: int = max(32, multiprocessing.cpu_count() - 1),
        log_file: str = "medical_tokenization.log"
    ):
        self.local_data_path = Path(local_data_path)
        self.local_data_path.mkdir(parents=True, exist_ok=True)
        (self.local_data_path / "tokens").mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.allowed_extensions = allowed_extensions
        self.allowed_mimetypes = allowed_mimetypes
        self.chunk_size = chunk_size
        self.processing_workers = processing_workers
        self.log_file = log_file
        self.image_processor = image_processor


###############################################################################
# Logging Setup
###############################################################################
def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

###############################################################################
# Tokenization Utilities
###############################################################################
class TokenizationUtilities:
    """
    Utilities for NLP preprocessing: padding, segment IDs, and masked LM inputs.
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
        """
        if not input_ids:
            raise ValueError("No input_ids provided for padding.")
        
        device = input_ids[0].device
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_masks):
            pad_length = max_length - len(ids)
            if padding_side == 'right':
                padded_ids = torch.cat([ids, torch.full((pad_length,), padding_value, dtype=ids.dtype)])
                padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
            else:
                padded_ids = torch.cat([torch.full((pad_length,), padding_value, dtype=ids.dtype), ids])
                padded_mask = torch.cat([torch.zeros(pad_length, dtype=mask.dtype), mask])
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        return (torch.stack(padded_input_ids).to(device), torch.stack(padded_attention_masks).to(device))

    @staticmethod
    def create_segment_ids(
        input_ids: Tensor,
        separator_token_id: int,
        cls_token_id: int
    ) -> Tensor:
        """
        Create segment IDs for multi-segment inputs (e.g., sentence pairs).
        """
        device = input_ids.device
        segment_ids = torch.zeros_like(input_ids).to(device)
        for i, seq in enumerate(input_ids):
            sep_positions = (seq == separator_token_id).nonzero(as_tuple=True)[0]
            cls_positions = (seq == cls_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0 and len(cls_positions) > 0:
                # For simplicity, assume only one CLS at start
                segment_ids[i, cls_positions[0]:sep_positions[0] + 1] = 0
                if sep_positions[0] + 1 < len(seq):
                    segment_ids[i, sep_positions[0] + 1:] = 1
        return segment_ids

    @staticmethod
    def generate_masked_lm_inputs(
        input_ids: Tensor,
        mask_probability: float = 0.15,
        mask_token_id: int = 103,
        special_token_ids: Optional[List[int]] = None,
        vocab_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate masked LM inputs using a standard BERT-style masking strategy:
        80% [MASK], 10% random, 10% original token.
        """
        if special_token_ids is None:
            special_token_ids = []
        
        device = input_ids.device
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        batch_size, seq_length = input_ids.shape

        # Identify maskable positions
        maskable_positions = torch.ones_like(input_ids, dtype=torch.bool)
        for sid in special_token_ids:
            maskable_positions &= (input_ids != sid)

        probabilities = torch.full(input_ids.shape, mask_probability, device=device)
        mask_probabilities = (torch.bernoulli(probabilities).bool()) & maskable_positions

        # 80% of the masked positions -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & mask_probabilities
        masked_input_ids[indices_replaced] = mask_token_id

        # 10% of the masked positions -> random tokens
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & mask_probabilities & ~indices_replaced
        if vocab_size is None:
            vocab_size = int(input_ids.max()) + 1
        random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=input_ids.dtype, device=device)
        masked_input_ids[indices_random] = random_tokens[indices_random]

        # The remaining 10% stay the same. 
        # For non-masked positions, set labels to -100 so they are ignored in loss
        labels[~mask_probabilities] = -100

        return masked_input_ids, labels, mask_probabilities


###############################################################################
# Hybrid Tokenization Strategy
###############################################################################
class HybridTokenizationStrategy:
    """
    Tokenization strategy supporting both autoregressive and bidirectional processing.
    """

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
    ) -> Dict[str, Tensor]:
        """
        Encode text for autoregressive (causal) tasks.
        """
        encodings = self.tokenizer.encode_batch(texts)
        input_ids_list, attention_masks = [], []
        for encoding in encodings:
            ids = encoding.ids
            if truncation:
                ids = ids[:max_length]
            tensor_ids = torch.tensor(ids, device=self.device)
            mask = torch.ones(len(tensor_ids), dtype=torch.long, device=self.device)
            input_ids_list.append(tensor_ids)
            attention_masks.append(mask)

        if padding:
            input_ids, attention_masks = self.utilities.dynamic_padding(input_ids_list, attention_masks)
        else:
            input_ids = torch.stack(input_ids_list)
            attention_masks = torch.stack(attention_masks)

        # Create causal masks
        causal_masks = []
        seq_lengths = attention_masks.sum(dim=1).tolist()
        for seq_len in seq_lengths:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float, device=self.device))
            pad_len = input_ids.shape[1] - seq_len
            # Pad the causal mask to fit the full sequence length
            causal_mask = torch.nn.functional.pad(causal_mask, (0, pad_len, 0, pad_len), value=0)
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
    ) -> Dict[str, Tensor]:
        """
        Encode text for bidirectional tasks.
        """
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        input_ids_list, attention_masks = [], []
        for encoding in encodings:
            ids = encoding.ids
            if truncation:
                ids = ids[:max_length]
            tensor_ids = torch.tensor(ids, device=self.device)
            mask = torch.ones(len(tensor_ids), dtype=torch.long, device=self.device)
            input_ids_list.append(tensor_ids)
            attention_masks.append(mask)

        if padding:
            input_ids, attention_masks = self.utilities.dynamic_padding(input_ids_list, attention_masks)
        else:
            input_ids = torch.stack(input_ids_list)
            attention_masks = torch.stack(attention_masks)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

    def hybrid_encode(
        self,
        texts: List[str],
        task_type: str = 'auto',
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Dynamically choose encoding strategy based on task type ('auto' or 'bi').
        """
        if task_type == 'auto':
            return self.autoregressive_encode(texts, **kwargs)
        elif task_type == 'bi':
            return self.bidirectional_encode(texts, **kwargs)
        else:
            raise ValueError(f"Invalid task type: {task_type}. Use 'auto' or 'bi'.")


###############################################################################
# MedicalTokenizer
###############################################################################
class MedicalTokenizer:
    """
    Tokenizer class for training and encoding with both autoregressive and bidirectional strategies.
    """

    def __init__(self, vocab_size: int = 60000, min_frequency: int = 2):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
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
        Train the tokenizer on the given files and save it.
        """
        save_path = Path(save_path)
        try:
            logging.info("Starting tokenizer training...")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.tokenizer.train(files, self.trainer)
            self.tokenizer.save(str(save_path))
            logging.info(f"Tokenizer saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to train/save tokenizer: {e}")
            raise

    def load(self, tokenizer_path: str):
        """
        Load a previously trained tokenizer.
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.strategy = HybridTokenizationStrategy(self.tokenizer)

    def encode(self, texts: List[str], task_type: str = 'auto', **kwargs) -> Dict[str, Tensor]:
        """
        Encode texts using the given task type ('auto' or 'bi').
        """
        return self.strategy.hybrid_encode(texts, task_type=task_type, **kwargs)


###############################################################################
# File Validator
###############################################################################
class FileValidator:
    """
    Validates and sanitizes file uploads.
    """
    def __init__(self, allowed_extensions: Set[str], allowed_mimetypes: Set[str], max_file_size: int):
        self.allowed_extensions = allowed_extensions
        self.allowed_mimetypes = allowed_mimetypes
        self.max_file_size = max_file_size

    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                return False, "File does not exist"
            if not file_path.is_file():
                return False, "Path is not a file"
            if file_path.suffix.lower() not in self.allowed_extensions:
                return False, f"Unsupported file extension. Allowed: {self.allowed_extensions}"
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type not in self.allowed_mimetypes:
                return False, f"Invalid file type: {mime_type}"
            if file_path.stat().st_size > self.max_file_size:
                return False, f"File too large. Maximum size: {self.max_file_size / 1024 / 1024}MB"
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        filename_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        ext = Path(filename).suffix
        safe_name = f"file_{filename_hash}{ext}"
        return safe_name


###############################################################################
# Dataset Processor
###############################################################################
class DatasetProcessor:
    """
    Enhanced dataset processor with robust text extraction, incremental processing,
    and sophisticated preprocessing capabilities.
    """
    def __init__(self, datasets: List[Dict[str, Any]], config: Config):
        self.datasets = datasets
        self.config = config
        self.file_validator = FileValidator(
            allowed_extensions=config.allowed_extensions,
            allowed_mimetypes=config.allowed_mimetypes,
            max_file_size=config.max_file_size
        )
        self.config.local_data_path.mkdir(parents=True, exist_ok=True)
        self.image_processor = ViTImageProcessor.from_pretrained(config.image_processor)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize preprocessing tools
        self.setup_preprocessing()
        self.temp_dir = tempfile.mkdtemp(prefix="dataset_processor_")
        self.retry_config = {
            'max_attempts': 3,
            'delay': 1,
            'max_delay': 10,
            'backoff': 2
        }
        
        # Add memory monitoring
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.initial_chunk_size = self.config.chunk_size

    def __del__(self):
        """Cleanup temporary directory on object destruction"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logging.error(f"Failed to cleanup temporary directory: {e}")

    @contextmanager
    def _managed_temp_file(self, suffix: str = '.txt') -> Generator[Path, None, None]:
        """Context manager for temporary file handling"""
        temp_path = Path(tempfile.mktemp(dir=self.temp_dir, suffix=suffix))
        try:
            yield temp_path
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _with_retry(self, func, *args, **kwargs):
        """Retry mechanism for operations that might fail transiently"""
        attempt = 0
        last_exception = None
        while attempt < self.retry_config['max_attempts']:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                attempt += 1
                if attempt < self.retry_config['max_attempts']:
                    delay = min(
                        self.retry_config['delay'] * (self.retry_config['backoff'] ** (attempt - 1)),
                        self.retry_config['max_delay']
                    )
                    logging.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
        
        logging.error(f"Operation failed after {attempt} attempts: {str(last_exception)}")
        raise last_exception

    def setup_preprocessing(self):
        """Initialize text preprocessing tools and configurations."""
        self.text_cleanup_patterns = {
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_numbers': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'extra_spaces': r'\s+',
            'special_chars': r'[^\w\s]'
        }
        
        self.dataset_specific_configs = {
            'medical': {
                'required_fields': ['diagnosis', 'symptoms', 'treatment'],
                'text_fields': ['notes', 'description'],
                'numeric_fields': ['age', 'dosage'],
                'date_fields': ['visit_date', 'follow_up']
            },
            'general': {
                'required_fields': ['text'],
                'text_fields': ['content', 'description'],
                'numeric_fields': [],
                'date_fields': []
            }
        }

    def extract_text(self, record: Any, dataset_type: str = 'general') -> str:
        """
        Enhanced text extraction with dataset-specific handling and validation.
        """
        try:
            if isinstance(record, str):
                return self.preprocess_text(record)
            
            if isinstance(record, dict):
                config = self.dataset_specific_configs.get(dataset_type, 
                                                        self.dataset_specific_configs['general'])
                
                # Extract and combine text from all relevant fields
                text_parts = []
                
                # Process required fields
                for field in config['required_fields']:
                    if field in record:
                        text_parts.append(str(record[field]))
                    else:
                        logging.warning(f"Required field '{field}' missing in record")
                
                # Process optional text fields
                for field in config['text_fields']:
                    if field in record:
                        text_parts.append(str(record[field]))
                
                # Process numeric fields with context
                for field in config['numeric_fields']:
                    if field in record:
                        text_parts.append(f"{field}: {record[field]}")
                
                # Process date fields with formatting
                for field in config['date_fields']:
                    if field in record:
                        try:
                            date_val = pd.to_datetime(record[field]).strftime('%Y-%m-%d')
                            text_parts.append(f"{field}: {date_val}")
                        except:
                            logging.warning(f"Could not parse date in field '{field}'")
                
                combined_text = " ".join(text_parts)
                return self.preprocess_text(combined_text)
            
            return ""
            
        except Exception as e:
            logging.error(f"Error in text extraction: {str(e)}")
            return ""

    def preprocess_text(self, text: str) -> str:
        """
        Apply sophisticated text preprocessing.
        """
        try:
            # Remove URLs
            text = re.sub(self.text_cleanup_patterns['urls'], '', text)
            
            # Remove email addresses
            text = re.sub(self.text_cleanup_patterns['emails'], '', text)
            
            # Remove phone numbers
            text = re.sub(self.text_cleanup_patterns['phone_numbers'], '', text)
            
            # Basic cleaning
            text = text.strip()
            text = re.sub(self.text_cleanup_patterns['extra_spaces'], ' ', text)
            
            # Remove non-printable characters
            text = ''.join(char for char in text if char.isprintable())
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logging.error(f"Error in text preprocessing: {str(e)}")
            return text

    def process_with_memory_management(self, texts: Generator[str, None, None]) -> Generator[str, None, None]:
        """Memory-efficient text processing using generators"""
        buffer_size = 1000
        buffer = []
        try:
            for text in texts:
                processed_text = self.preprocess_text(text)
                if processed_text.strip():
                    buffer.append(processed_text)
                    if len(buffer) >= buffer_size:
                        yield from buffer
                        buffer.clear()
                        gc.collect()  # Force garbage collection
        except Exception as e:
            logging.error(f"Error in text processing stream: {e}")
        finally:
            if buffer:
                yield from buffer

    @contextmanager
    def _memory_guard(self, threshold: float = 0.9):
        """
        Context manager to monitor and manage memory usage.
        
        Args:
            threshold: Memory usage threshold (0-1) to trigger cleanup
        """
        try:
            yield
        finally:
            memory = psutil.virtual_memory()
            if memory.percent > threshold * 100:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _get_optimal_chunk_size(self) -> int:
        """
        Dynamically calculate optimal chunk size based on system resources.
        """
        memory = psutil.virtual_memory()
        cpu_count = multiprocessing.cpu_count()
        
        # Base size on available memory (assuming 8KB per text)
        mem_based_size = max(1000, int(memory.available * 0.1 / 8192))
        
        # Adjust based on CPU cores
        cpu_based_size = max(1000, self.initial_chunk_size // cpu_count)
        
        return min(mem_based_size, cpu_based_size, self.config.chunk_size)

    def _get_optimal_workers(self) -> int:
        """Dynamically adjust worker count based on system resources"""
        cpu_count = multiprocessing.cpu_count()
        memory = psutil.virtual_memory()
        
        # Reduce workers if memory usage is high
        if memory.percent > self.memory_threshold * 100:
            return max(1, cpu_count // 2)
        return self.config.processing_workers

    def _process_chunk(self, chunk: List[str], worker_id: int) -> List[str]:
        """
        Process a chunk of texts with improved error handling and memory management.
        
        Args:
            chunk: List of texts to process
            worker_id: ID of the worker processing this chunk
            
        Returns:
            List[str]: List of processed texts
        """
        processed = []
        try:
            # Monitor memory usage
            if psutil.virtual_memory().percent > self.memory_threshold * 100:
                logging.warning(f"Worker {worker_id}: High memory usage detected")
                gc.collect()
            
            for idx, text in enumerate(chunk):
                try:
                    if not isinstance(text, str):
                        continue
                        
                    processed_text = self.preprocess_text(text)
                    if processed_text.strip():
                        processed.append(processed_text)
                        
                except Exception as e:
                    logging.error(f"Worker {worker_id}: Error processing text at index {idx}: {e}")
                    continue
                
        except Exception as e:
            logging.error(f"Worker {worker_id}: Chunk processing error: {e}")
            
        finally:
            # Clean up memory
            chunk.clear()
            gc.collect()
            
        return processed

    def _process_generic_texts(
        self,
        texts: Generator[str, None, None],
        output_path: Path,
        chunk_size: Optional[int] = None,
        timeout: int = 300
    ) -> Optional[str]:
        """
        Process texts using a memory-efficient generator approach.
        
        Args:
            texts: Generator yielding texts to process
            output_path: Path to save processed texts
            chunk_size: Optional custom chunk size (default: None, uses dynamic sizing)
            timeout: Timeout in seconds for each worker (default: 300)
        """
        if chunk_size is None:
            chunk_size = self._get_optimal_chunk_size()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                buffer = []
                for text in texts:
                    buffer.append(text)
                    
                    if len(buffer) >= chunk_size:
                        try:
                            workers = self._get_optimal_workers()
                            with ProcessPoolExecutor(max_workers=workers) as executor:
                                futures = []
                                
                                # Split buffer into sub-chunks
                                for worker_id, sub_chunk in enumerate(split_chunk(buffer, workers)):
                                    futures.append(
                                        executor.submit(
                                            self._with_retry,
                                            self._process_chunk,
                                            sub_chunk,
                                            worker_id
                                        )
                                    )
                                
                                # Process results with timeout
                                for future in as_completed(futures):
                                    try:
                                        result = future.result(timeout=timeout)
                                        if result:
                                            f.write('\n'.join(result) + '\n')
                                            f.flush()
                                    except TimeoutError:
                                        logging.error(f"Worker timeout after {timeout} seconds")
                                    except Exception as e:
                                        logging.error(f"Error processing chunk: {e}")
                    
                        finally:
                            buffer.clear()
                            gc.collect()
                
                # Process remaining texts
                if buffer:
                    try:
                        processed = self._process_chunk(buffer, worker_id=0)
                        if processed:
                            f.write('\n'.join(processed) + '\n')
                    except Exception as e:
                        logging.error(f"Error processing final chunk: {e}")
            
            return str(output_path)
        
        except Exception as e:
            logging.error(f"Error in _process_generic_texts: {e}")
            if output_path.exists():
                output_path.unlink()
            return None

    def process(self) -> List[str]:
        """
        Process all datasets and return a list of processed file paths.
        For Hugging Face datasets, load and process them.
        For local files, just process them directly.
        """
        processed_files = []

        for dataset_info in self.datasets:
            # Determine if it's a Hugging Face dataset or a local file
            if "name" in dataset_info and not dataset_info.get("path"):
                # Hugging Face dataset
                processed_files.extend(self._process_hf_dataset(dataset_info))
            elif "path" in dataset_info:
                # Local file
                file_path = dataset_info["path"]
                processed = self._process_local_file(file_path)
                if processed is not None:
                    processed_files.append(processed)
            else:
                logging.warning(f"Dataset info not recognized: {dataset_info}")

        return processed_files

    def _process_hf_dataset(self, dataset_info: Dict[str, Any]) -> List[str]:
        """
        Process a Hugging Face dataset (download and iterate over splits).
        """
        name = dataset_info["name"]
        download_config = dataset_info.get("download_config", {})
        processed_files = []
        
        try:
            # Ensure cache_dir is set
            cache_dir = download_config.get("cache_dir", "./hf_cache")
            download_config["cache_dir"] = cache_dir

            # Load dataset with streaming
            dataset = load_dataset(name, **download_config)
            processed = self._process_hf_split(dataset, f"{name}_train")
            if processed:
                processed_files.append(processed)

        except Exception as e:
            logging.error(f"Error processing dataset {name}: {str(e)}")
            logging.error(traceback.format_exc())
            return []

        return processed_files

    def _process_hf_split(self, hf_dataset, prefix: str) -> Optional[str]:
        """
        Process a single split of a Hugging Face dataset.
        """
        output_file = self.config.local_data_path / f"processed_{prefix}.txt"
        try:
            # Process streaming dataset in batches
            records = []
            batch = []
            skipped_count = 0
            for x in hf_dataset:
                text = self.extract_text(x)
                if not text.strip():
                    skipped_count += 1
                    continue
                if text.strip():
                    batch.append(text)
                    if len(batch) >= self.config.chunk_size:
                        # Process the current batch
                        processed_texts = self._parallel_process_texts(batch, self.config.chunk_size)
                        records.extend(processed_texts)
                        batch = []  # Reset batch

            # Process any remaining items
            if batch:
                processed_texts = self._parallel_process_texts(batch, self.config.chunk_size)
                records.extend(processed_texts)

            # Write all processed texts to file
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in records:
                    f.write(text + "\n")

            logging.info(f"Skipped {skipped_count} invalid records.")

            return str(output_file)
        except Exception as e:
            logging.error(f"Error processing HF dataset {prefix}: {e}")
            if output_file.exists():
                output_file.unlink()
            return None

    def _process_local_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Process a single local file (CSV, JSONL, TXT, or image) and return output path.
        """
        file_path = Path(file_path)
        is_valid, error_msg = self.file_validator.validate_file(file_path)
        if not is_valid:
            logging.error(f"File validation failed for {file_path}: {error_msg}")
            return None

        safe_filename = self.file_validator.sanitize_filename(file_path.name)
        output_file = self.config.local_data_path / f"processed_{safe_filename}"

        try:
            ext = file_path.suffix.lower()
            # Add image processing logic
            if ext in {'.jpg', '.jpeg', '.png', '.webp'}:
                return self._process_image_file(file_path, output_file.with_suffix('.json'))
            elif ext == '.csv':
                return self._process_csv_file(file_path, output_file)
            elif ext == '.jsonl':
                return self._process_jsonl_file(file_path, output_file)
            elif ext in {'.txt', '.text'}:
                return self._process_text_file(file_path, output_file)
            elif ext == '.json':
                # For .json, assume it's a list of objects
                return self._process_json_file(file_path, output_file)
            else:
                logging.warning(f"Unsupported file format: {ext}")
                return None
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            if output_file.exists():
                output_file.unlink()
            return None

    def _process_csv_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """Process CSV files with improved memory management and error handling."""
        try:
            # Calculate optimal chunk size based on available memory
            total_memory = psutil.virtual_memory().total
            chunk_size = min(self.config.chunk_size, 
                            max(1000, int(total_memory * 0.1 / 8192)))  # Assuming 8KB per text
            
            texts = []
            for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                try:
                    # Process each record in the chunk
                    chunk_texts = []
                    for _, record in chunk.iterrows():
                        try:
                            text = self.extract_text(record.to_dict())
                            if text.strip():
                                chunk_texts.append(text)
                        except Exception as e:
                            logging.warning(f"Error processing record: {e}")
                            continue
                    
                    # Process accumulated texts
                    if chunk_texts:
                        result = _process_generic_texts(
                            chunk_texts,
                            output_path.with_suffix(f'.part{len(texts)}'),
                            self.config.chunk_size,
                            self.config.processing_workers
                        )
                        if result:
                            texts.append(result)
                
                except Exception as e:
                    logging.error(f"Error processing chunk: {e}")
                    continue
            
            # Merge all part files
            if texts:
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    for part_file in texts:
                        try:
                            with open(part_file, 'r', encoding='utf-8') as infile:
                                shutil.copyfileobj(infile, outfile)
                            os.remove(part_file)  # Clean up part file
                        except Exception as e:
                            logging.error(f"Error merging part file {part_file}: {e}")
                
                return str(output_path)
            
            return None
            
        except Exception as e:
            logging.error(f"Error processing CSV file {input_path}: {e}")
            return None

    def _process_jsonl_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """Process JSONL files using the generic text processor."""
        try:
            texts = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        text = self.extract_text(obj)
                        if text.strip():
                            texts.append(text)
                            
                        # Process in chunks to manage memory
                        if len(texts) >= self.config.chunk_size:
                            return _process_generic_texts(
                                texts,
                                output_path,
                                self.config.chunk_size,
                                self.config.processing_workers
                            )
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in {input_path}")
                        continue
                    
            # Process any remaining texts
            if texts:
                return _process_generic_texts(
                    texts,
                    output_path,
                    self.config.chunk_size,
                    self.config.processing_workers
                )
            return None
            
        except Exception as e:
            logging.error(f"Error processing JSONL file {input_path}: {str(e)}")
            return None

    def _process_json_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """Process JSON files using the generic text processor."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logging.error(f"JSON file {input_path} must contain a list of objects")
                return None
            
            texts = [self.extract_text(item) for item in data]
            texts = [text for text in texts if text.strip()]
            
            if not texts:
                logging.warning(f"No valid texts found in {input_path}")
                return None
            
            return _process_generic_texts(
                texts,
                output_path,
                self.config.chunk_size,
                self.config.processing_workers
            )
            
        except Exception as e:
            logging.error(f"Error processing JSON file {input_path}: {str(e)}")
            return None

    def _process_text_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """Process text files using the generic text processor."""
        try:
            texts = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip()
                    if text:
                        texts.append(text)
                        
                    # Process in chunks to manage memory
                    if len(texts) >= self.config.chunk_size:
                        return _process_generic_texts(
                            texts,
                            output_path,
                            self.config.chunk_size,
                            self.config.processing_workers
                        )
                    
            # Process any remaining texts
            if texts:
                return _process_generic_texts(
                    texts,
                    output_path,
                    self.config.chunk_size,
                    self.config.processing_workers
                )
            return None
            
        except Exception as e:
            logging.error(f"Error processing text file {input_path}: {str(e)}")
            return None

    def _process_image_file(self, input_path: Path, output_path: Path, model_type: str = 'ViT') -> Optional[str]:
        """
        Process image files by extracting features and converting to text representation.
        """
        try:
            image = Image.open(input_path).convert('RGB')
            if model_type == 'ViT':
                inputs = self.image_processor(image, return_tensors="pt")
            elif model_type == 'CLIP':
                inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            # Further processing based on model_type
        except Exception as e:
            logging.error(f"Error processing image file {input_path}: {str(e)}")
            return None

    def _parallel_process_texts(self, texts: List[str], chunk_size: int) -> List[str]:
        """
        Process a list of texts in parallel chunks with improved error handling.
        """
        processed_texts = []
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            try:
                with ThreadPoolExecutor(max_workers=self.config.processing_workers) as executor:
                    sub_chunk_size = max(100, chunk_size // self.config.processing_workers)
                    futures = []
                    
                    for j in range(0, len(chunk), sub_chunk_size):
                        sub_chunk = chunk[j:j + sub_chunk_size]
                        futures.append(
                            executor.submit(
                                self._process_chunk, 
                                sub_chunk, 
                                worker_id=j // sub_chunk_size
                            )
                        )
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                processed_texts.extend(result)
                        except Exception as e:
                            logging.error(f"Error in chunk processing: {str(e)}")
                            continue
                        
            except Exception as e:
                logging.error(f"Error in parallel processing: {str(e)}")
                continue
            
            gc.collect()
        
        return processed_texts


###############################################################################
# Main Execution
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    process_parser = subparsers.add_parser('process')
    train_parser = subparsers.add_parser('train')

    parser.add_argument('--local_data_path', type=str, default="C:\\Users\\ASUS\\Desktop\\LuminaLM\\Data", help="Path to store processed data and tokenizer.")
    parser.add_argument('--vocab_size', type=int, default=60000, help="Vocabulary size for the tokenizer.")
    parser.add_argument('--min_frequency', type=int, default=3, help="Minimum frequency for BPE merges.")
    parser.add_argument('--log_file', type=str, default="medical_tokenization.log", help="Log file name.")
    parser.add_argument('--cache_dir', type=str, default=".data_cache", help="Cache directory for datasets.")
    parser.add_argument('--verbosity', type=str, default='INFO', help="Logging verbosity level.")
    parser.add_argument('--retry_max_attempts', type=int, default=3, help="Maximum retry attempts.")
    parser.add_argument('--retry_delay', type=int, default=1, help="Initial delay between retries.")
    parser.add_argument('--retry_max_delay', type=int, default=10, help="Maximum delay between retries.")
    parser.add_argument('--retry_backoff', type=int, default=2, help="Backoff multiplier for retries.")
    parser.add_argument('--datasets', type=str, nargs='+', help="List of datasets to process.")
    parser.add_argument('--file_formats', type=str, nargs='+', help="List of file formats to process.")
    parser.add_argument('--resume', action='store_true', help="Resume processing")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.verbosity.upper(), None))

    # Updated dataset configuration
    datasets = [
    {
        "name": "openwebtext",
        "download_config": {
            "cache_dir": args.cache_dir,
            "streaming": True,
            "split": "train",
            "trust_remote_code": True
        }
    }
]
    
    try:
        # Create necessary directories
        tokens_dir = Path(args.local_data_path) / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        config = Config(
            local_data_path=args.local_data_path,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            log_file=args.log_file
        )

        dataset_processor = DatasetProcessor(datasets, config)
        tokenizer = MedicalTokenizer(vocab_size=config.vocab_size, min_frequency=config.min_frequency)

        # Process and train
        logging.info("Starting dataset processing...")
        processed_files = dataset_processor.process()
        if not processed_files:
            raise ValueError("No files were successfully processed")
        logging.info(f"Dataset processing completed. Processed {len(processed_files)} files.")
        
        tokenizer_path = tokens_dir / "Medical_tokenizer.json"
        tokenizer.train(processed_files, str(tokenizer_path))
        logging.info("Tokenizer training completed successfully.")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

def _process_generic_texts(texts: List[str], output_path: Path, chunk_size: int, num_workers: int) -> Optional[str]:
    """
    Process a list of texts with proper chunking and error handling.
    
    Args:
        texts: List of texts to process
        output_path: Path to save processed texts
        chunk_size: Size of chunks for processing
        num_workers: Number of workers for parallel processing
        
    Returns:
        Optional[str]: Path to processed file if successful, None otherwise
    """
    try:
        # Process in chunks to manage memory
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                try:
                    # Process chunk
                    processed_chunk = []
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        futures = [
                            executor.submit(
                                DatasetProcessor._process_chunk, 
                                chunk[j:j + chunk_size // num_workers],
                                worker_id=j
                            ) 
                            for j in range(0, len(chunk), chunk_size // num_workers)
                        ]
                        
                        for future in as_completed(futures):
                            try:
                                processed_chunk.extend(future.result())
                            except Exception as e:
                                logging.error(f"Error processing chunk: {e}")
                                continue
                    
                    # Write chunk with error handling
                    if processed_chunk:
                        try:
                            f.write('\n'.join(processed_chunk) + '\n')
                            f.flush()
                        except IOError as e:
                            logging.error(f"Error writing chunk to file: {e}")
                            # Attempt to write to backup file
                            backup_path = output_path.with_suffix('.backup')
                            with open(backup_path, 'a', encoding='utf-8') as backup_f:
                                backup_f.write('\n'.join(processed_chunk) + '\n')
                
                except Exception as e:
                    logging.error(f"Error processing chunk {i//chunk_size}: {e}")
                    continue
                
                # Clear memory
                processed_chunk.clear()
        
        return str(output_path)
    
    except Exception as e:
        logging.error(f"Error in _process_generic_texts: {e}")
        return None

def dynamic_chunk_size():
    total_memory = psutil.virtual_memory().total
    return max(1000, int(total_memory * 0.1 / 8192))  # Assuming 8KB per text

chunk_size = dynamic_chunk_size()

class TestUtils(unittest.TestCase):
    def test_with_retry(self):
        # Add test cases for with_retry
        pass

    def test_managed_temp_file(self):
        # Add test cases for managed_temp_file
        pass

    def test_split_chunk(self):
        # Add test cases for split_chunk
        pass

if __name__ == '__main__':
    unittest.main()
