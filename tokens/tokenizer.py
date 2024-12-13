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
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Generator, Callable
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
import asyncio
import random
import yaml
from tqdm.contrib.concurrent import thread_map
import threading
import aiofiles
from logging.handlers import RotatingFileHandler
import chardet

###############################################################################
# Configuration
###############################################################################
class Config:
    """Configuration class for tokenizer training."""
    def __init__(
        self,
        local_data_path: str,
        vocab_size: int = 60000,
        min_frequency: int = 2,
        log_file: str = "tokenizer.log",
        chunk_size: int = 1000,
        max_workers: int = None,
        memory_threshold: float = 0.8,
        allowed_extensions: Set[str] = None,
        allowed_mimetypes: Set[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default
        gpu_memory_threshold: float = 0.8
    ):
        self.local_data_path = Path(local_data_path)
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.log_file = log_file
        self.chunk_size = chunk_size
        self.max_workers = max_workers if max_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        self.memory_threshold = memory_threshold
        self.allowed_extensions = allowed_extensions or {'.txt', '.json', '.jsonl', '.csv'}
        self.allowed_mimetypes = allowed_mimetypes or {'text/plain', 'application/json', 'text/csv'}
        self.max_file_size = max_file_size
        self.gpu_memory_threshold = gpu_memory_threshold

    @property
    def processing_workers(self) -> int:
        """Get the number of workers based on system resources."""
        if self.max_workers == 0:  # Auto-configure
            cpu_count = multiprocessing.cpu_count()
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                return 1
            elif memory.percent > 80:
                return max(1, cpu_count // 4)
            elif memory.percent > 70:
                return max(1, cpu_count // 2)
            else:
                return max(1, cpu_count - 1)
        
        return self.max_workers


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
        Dynamically pad input sequences with improved validation.
        """
        if not input_ids:
            raise ValueError("No input_ids provided for padding.")
        
        # Validate input dimensions
        if any(len(ids.shape) != 1 for ids in input_ids):
            raise ValueError("All input_ids must be 1-dimensional tensors")
        
        # Calculate safe max_length
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)
        else:
            actual_max = max(len(ids) for ids in input_ids)
            if actual_max > max_length:
                logging.warning(f"Truncating sequences from {actual_max} to {max_length}")
        
        device = input_ids[0].device
        padded_input_ids = []
        padded_attention_masks = []

        try:
            for ids, mask in zip(input_ids, attention_masks):
                if len(ids) != len(mask):
                    raise ValueError("Mismatched lengths between input_ids and attention_mask")
                
                pad_length = max_length - len(ids)
                if padding_side == 'right':
                    padded_ids = torch.cat([ids[:max_length], 
                                          torch.full((max(0, pad_length),), padding_value, dtype=ids.dtype)])
                    padded_mask = torch.cat([mask[:max_length], 
                                           torch.zeros(max(0, pad_length), dtype=mask.dtype)])
                else:
                    padded_ids = torch.cat([torch.full((max(0, pad_length),), padding_value, dtype=ids.dtype),
                                          ids[:max_length]])
                    padded_mask = torch.cat([torch.zeros(max(0, pad_length), dtype=mask.dtype),
                                           mask[:max_length]])
                
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            return (torch.stack(padded_input_ids).to(device), 
                    torch.stack(padded_attention_masks).to(device))
                
        except Exception as e:
            raise RuntimeError(f"Error during padding: {str(e)}")

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

    @staticmethod
    def validate_inputs(input_ids: List[Tensor], attention_masks: Optional[List[Tensor]] = None) -> None:
        """Validate tokenizer inputs."""
        if not input_ids:
            raise ValueError("Empty input_ids provided")
            
        if attention_masks is not None and len(input_ids) != len(attention_masks):
            raise ValueError(f"Mismatched lengths: {len(input_ids)} input_ids vs {len(attention_masks)} attention_masks")
            
        shapes = [ids.shape for ids in input_ids]
        if not all(len(shape) == 1 for shape in shapes):
            raise ValueError("All input_ids must be 1-dimensional tensors")

    @staticmethod
    def validate_special_tokens(
        special_tokens: Dict[str, int],
        required_tokens: Set[str] = {'pad', 'unk', 'mask'}
    ) -> None:
        """Validate special token configuration."""
        missing = required_tokens - set(special_tokens.keys())
        if missing:
            raise ValueError(f"Missing required special tokens: {missing}")

    @staticmethod
    def create_attention_mask(
        input_ids: Tensor,
        padding_token_id: int,
        dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Create attention mask with proper padding handling."""
        return (input_ids != padding_token_id).to(dtype)

    @staticmethod
    def create_causal_mask(size: int, dtype: torch.dtype = torch.float32) -> Tensor:
        """Create causal attention mask with proper type handling."""
        return torch.triu(torch.ones(size, size, dtype=dtype) * float('-inf'), diagonal=1)


###############################################################################
# Memory Management
###############################################################################
class MemoryManager:
    """Manages memory usage and cleanup."""
    def __init__(self):
        self.threshold = 0.9

    def clean_memory(self, aggressive: bool = False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.ipc_collect()

    @contextmanager
    def monitor_memory(self, operation: str = ""):
        try:
            yield
        finally:
            if self.should_cleanup():
                self.clean_memory()

    def should_cleanup(self) -> bool:
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            if memory_reserved > 0:
                return (memory_allocated / memory_reserved) > self.threshold
        return False


###############################################################################
# Hybrid Tokenization Strategy
###############################################################################
class HybridTokenizationStrategy:
    """
    Tokenization strategy supporting both autoregressive and bidirectional processing.
    """

    def __init__(self, tokenizer: Tokenizer, memory_manager: Optional[MemoryManager] = None):
        self.tokenizer = tokenizer
        self.memory_manager = memory_manager or MemoryManager()
        self.utils = TokenizationUtilities()

    def encode(
        self,
        texts: List[str],
        task_type: str = 'auto',
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Enhanced encode method with task-specific optimizations.
        """
        if not texts:
            raise ValueError("Empty texts provided for encoding")

        # Determine encoding strategy
        if task_type == 'auto':
            # Analyze text to determine best strategy
            avg_length = sum(len(text.split()) for text in texts) / len(texts)
            task_type = 'bi' if avg_length < 512 else 'auto'  # Use bidirectional for shorter texts

        try:
            with self.memory_manager.monitor_memory(f"{task_type} encoding"):
                if task_type == 'auto':
                    return self.autoregressive_encode(texts, max_length, **kwargs)
                else:
                    return self.bidirectional_encode(texts, max_length, **kwargs)
        except Exception as e:
            logging.error(f"Encoding failed for task_type {task_type}: {str(e)}")
            raise

    def autoregressive_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Autoregressive encoding with memory-efficient batching.
        """
        try:
            chunk_manager = ChunkManager(self.memory_manager)
            encoded_chunks = []
            
            for text_chunk in chunk_manager.chunk_iterator(texts):
                # Encode chunk
                encodings = self.tokenizer.encode_batch(text_chunk)
                
                # Process encodings
                input_ids = [torch.tensor(enc.ids) for enc in encodings]
                attention_mask = [torch.tensor(enc.attention_mask) for enc in encodings]
                
                # Validate and pad
                self.utils.validate_inputs(input_ids, attention_mask)
                padded_ids, padded_mask = TokenizationUtilities.dynamic_padding(
                    input_ids, attention_mask, max_length=max_length
                )
                
                # Create causal mask
                causal_mask = self.utils.create_causal_mask(padded_ids.size(1))
                
                encoded_chunks.append({
                    'input_ids': padded_ids,
                    'attention_mask': padded_mask,
                    'causal_mask': causal_mask
                })
            
            # Combine chunks
            return self._combine_encoded_chunks(encoded_chunks)
            
        except Exception as e:
            logging.error(f"Autoregressive encoding failed: {str(e)}")
            raise

    def bidirectional_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Bidirectional encoding with memory-efficient batching.
        """
        try:
            chunk_manager = ChunkManager(self.memory_manager)
            encoded_chunks = []
            
            for text_chunk in chunk_manager.chunk_iterator(texts):
                # Encode chunk
                encodings = self.tokenizer.encode_batch(text_chunk)
                
                # Process encodings
                input_ids = [torch.tensor(enc.ids) for enc in encodings]
                attention_mask = [torch.tensor(enc.attention_mask) for enc in encodings]
                
                # Validate and pad
                self.utils.validate_inputs(input_ids, attention_mask)
                padded_ids, padded_mask = TokenizationUtilities.dynamic_padding(
                    input_ids, attention_mask, max_length=max_length
                )
                
                encoded_chunks.append({
                    'input_ids': padded_ids,
                    'attention_mask': padded_mask
                })
            
            # Combine chunks
            return self._combine_encoded_chunks(encoded_chunks)
            
        except Exception as e:
            logging.error(f"Bidirectional encoding failed: {str(e)}")
            raise

    def _combine_encoded_chunks(
        self,
        chunks: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """
        Combine encoded chunks with enhanced validation and memory efficiency.
        """
        if not chunks:
            raise ValueError("No chunks to combine")
            
        try:
            # Validate chunk compatibility before combining
            reference_shapes = {key: chunks[0][key].shape[1:] for key in chunks[0].keys()}
            for i, chunk in enumerate(chunks):
                if set(chunk.keys()) != set(reference_shapes.keys()):
                    raise ValueError(f"Mismatched keys in chunk {i}")
                for key, shape in reference_shapes.items():
                    if chunk[key].shape[1:] != shape:
                        raise ValueError(
                            f"Mismatched shapes for key '{key}' in chunk {i}: "
                            f"expected {shape}, got {chunk[key].shape[1:]}"
                        )

            # Combine chunks with memory monitoring
            combined = {}
            for key in chunks[0].keys():
                tensors = [chunk[key] for chunk in chunks]
                
                # Calculate total memory requirement
                total_elements = sum(t.numel() for t in tensors)
                element_size = tensors[0].element_size()
                required_memory = total_elements * element_size * 2  # Factor of 2 for safety
                
                # Check available memory
                if torch.cuda.is_available():
                    available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                    if required_memory > available_memory * 0.9:  # 90% threshold
                        # Fall back to CPU concatenation
                        tensors = [t.cpu() for t in tensors]
                        combined[key] = torch.cat(tensors, dim=0).to(chunks[0][key].device)
                    else:
                        combined[key] = torch.cat(tensors, dim=0)
                else:
                    combined[key] = torch.cat(tensors, dim=0)
                
                # Clear intermediate tensors
                del tensors
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            return combined
            
        except Exception as e:
            logging.error(f"Failed to combine encoded chunks: {str(e)}")
            raise

    def _get_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on system resources"""
        available_memory = psutil.virtual_memory().available
        base_chunk_size = 1000  # Minimum chunk size
        
        # Use 5% of available memory, assuming 8KB per text
        memory_based_size = max(base_chunk_size, int(available_memory * 0.05 / 8192))
        
        # Cap at a reasonable maximum
        max_chunk_size = 100000
        return min(memory_based_size, max_chunk_size)

    def _get_optimal_workers(self, data_size: int) -> int:
        """
        Dynamically determine optimal number of workers based on data size and system resources.
        """
        cpu_count = multiprocessing.cpu_count()
        
        # For small datasets, limit parallelization
        if data_size < 1000:
            return min(2, cpu_count)
        elif data_size < 10000:
            return min(cpu_count // 2, 4)
            
        # For larger datasets, consider memory and CPU
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 80:
            return max(1, cpu_count // 4)
        elif memory_usage > 60:
            return max(2, cpu_count // 2)
        
        return max(1, cpu_count - 1)


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
        self.path_manager = TokenizerPathManager(Path.cwd())

    def train(self, files: List[str], save_path: str):
        """Train the tokenizer with enhanced path validation."""
        try:
            logging.info("Starting tokenizer training...")
            
            # Validate save path before training
            save_path = self.path_manager.validate_save_path(save_path)
            
            # Train tokenizer
            self.tokenizer.train(files, self.trainer)
            
            # Save with backup handling
            self.path_manager.safe_save(self.tokenizer, save_path)
            
        except Exception as e:
            logging.error(f"Failed to train/save tokenizer: {e}")
            raise

    def load(self, tokenizer_path: str):
        """Load a previously trained tokenizer with validation."""
        path = Path(tokenizer_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Tokenizer path is not a file: {path}")
        
        try:
            # Validate file format
            with open(path, 'r') as f:
                config = json.load(f)
                if not all(key in config for key in ['model', 'vocab', 'merges']):
                    raise ValueError("Invalid tokenizer file format")
            
            self.tokenizer = Tokenizer.from_file(str(path))
            self.strategy = HybridTokenizationStrategy(self.tokenizer)
            logging.info(f"Successfully loaded tokenizer from {path}")
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in tokenizer file: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")

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
    """Enhanced dataset processor with robust text extraction and preprocessing."""
    
    def __init__(self, datasets: List[Dict[str, Any]], config: Config):
        self.datasets = datasets
        self.config = config
        self.memory_threshold = 0.8
        self.file_validator = FileValidator(
            allowed_extensions=config.allowed_extensions,
            allowed_mimetypes=config.allowed_mimetypes,
            max_file_size=config.max_file_size
        )
        
        # Initialize dataset-specific configurations with field definitions
        self.dataset_specific_configs = {
            'openwebtext': {
                'required_fields': ['text'],
                'text_fields': ['text', 'title'],
                'numeric_fields': [],
                'date_fields': [],
                'preprocessing': {
                    'lower_case': True,
                    'strip_newlines': True,
                    'min_length': 50
                }
            },
            'wikipedia': {
                'required_fields': ['text'],
                'text_fields': ['text', 'title', 'section'],
                'numeric_fields': ['id'],
                'date_fields': ['timestamp'],
                'preprocessing': {
                    'lower_case': False,
                    'strip_newlines': False,
                    'min_length': 100
                }
            },
            'general': {
                'required_fields': ['text'],
                'text_fields': ['text'],
                'numeric_fields': [],
                'date_fields': [],
                'preprocessing': {
                    'lower_case': True,
                    'strip_newlines': True,
                    'min_length': 20
                }
            }
        }
        
        # Initialize other attributes
        self.total_files = self._calculate_total_files()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.gpu_memory_monitor = GPUMemoryMonitor(threshold=config.gpu_memory_threshold)
        
        self.memory_monitor = MemoryMonitor(threshold=0.7)
        self.batch_size = self._calculate_optimal_batch_size()
        self.current_workers = self._calculate_optimal_workers()
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.current_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.current_workers)
        
        self.retry_config = {
            'max_attempts': 3,
            'delay': 1,
            'max_delay': 10,
            'backoff': 2
        }
        
        self.temp_dir = tempfile.mkdtemp()

    def _calculate_optimal_workers(self) -> int:
        """
        Dynamically determine optimal number of workers based on system resources and dataset size.
        """
        try:
            # Get system information
            cpu_count = multiprocessing.cpu_count()
            memory = psutil.virtual_memory()
            base_workers = max(1, cpu_count // 2)
            if memory.percent > 80:
                return max(1, cpu_count // 4)
            elif memory.percent > 60:
                return max(1, cpu_count // 2)
            
            #Adjust based on dataset size
            if self.total_files > 1000:
                base_workers = min(base_workers, 2)
            elif self.total_files > 1000:
                base_workers = min(base_workers, 4)
            
            #Consider GPU if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_usage = torch.cuda.memory_allocated()
                if gpu_memory_usage > gpu_memory * 0.8:
                    base_workers = max(1, base_workers // 2)

            if self.config.max_workers is not None:
                base_workers = min(base_workers, self.config.max_workers)
            
            return max(1, base_workers)
        except Exception as e:
            logging.error(f"Error calculating optimal workers: {str(e)}, falling back to {base_workers} workers")
            return max(1, multiprocessing.cpu_count() // 2)
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available system resources."""
        #Get available system memory
        memory = psutil.virtual_memory()
        available_memory = memory.available

        #Base batch size calculation(assuming ~1KB per text item)
        base_batch_size = max(100, available_memory // (1024 * 1024))

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_factor = (gpu_memory - gpu_memory_used) / gpu_memory
        
        cpu_count = multiprocessing.cpu_count()
        base_size = min(base_size, cpu_count * 100)

        min_batch_size = 100
        max_batch_size = 10000

        return max(min_batch_size, min(base_size, max_batch_size))
        
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for specific dataset with fallback to 'general'."""
        if dataset_name not in self.dataset_specific_configs:
            logging.warning(f"Unsupported dataset type: {dataset_name}, falling back to 'general'")
        return self.dataset_specific_configs.get(dataset_name, self.dataset_specific_configs['general'])
    
    def preprocess_text(self, text: str, dataset_name: str) -> str:
        """Apply dataset-specific preprocessing."""
        config = self.get_dataset_config(dataset_name)
        preprocessing = config['preprocessing']
        
        if not text:
            return ""
            
        if preprocessing.get('lower_case', True):
            text = text.lower()
            
        if preprocessing.get('strip_newlines', True):
            text = ' '.join(text.split())
            
        # Skip texts that are too short
        if len(text) < preprocessing.get('min_length', 20):
            return ""
            
        return text

    def extract_text(self, record: Any, dataset_type: str = 'general') -> str:
        """
        Enhanced text extraction with dataset-specific handling and validation.
        """
        try:
            config = self.get_dataset_config(dataset_type)
            
            if isinstance(record, str):
                return self.preprocess_text(record, dataset_type)
            
            if isinstance(record, dict):
                # Extract and combine text from all relevant fields
                text_parts = []
                missing_required = []
                
                # Process required fields
                for field in config['required_fields']:
                    if field in record:
                        value = record[field]
                        if value is not None and str(value).strip():
                            text_parts.append(str(value))
                    else:
                        missing_required.append(field)
                
                if missing_required:
                    logging.warning(
                        f"Missing required fields for {dataset_type} dataset: {missing_required}"
                    )
                    if not text_parts:  # If no required fields were found
                        return ""
                
                # Process optional text fields
                for field in config['text_fields']:
                    if field in record and record[field] is not None:
                        text_parts.append(str(record[field]))
                
                # Process numeric fields with context
                for field in config['numeric_fields']:
                    if field in record and record[field] is not None:
                        try:
                            numeric_value = float(record[field])
                            text_parts.append(f"{field}: {numeric_value}")
                        except (ValueError, TypeError):
                            logging.warning(f"Invalid numeric value in field '{field}'")
                
                # Process date fields with formatting
                for field in config['date_fields']:
                    if field in record and record[field] is not None:
                        try:
                            date_val = pd.to_datetime(record[field]).strftime('%Y-%m-%d')
                            text_parts.append(f"{field}: {date_val}")
                        except Exception as e:
                            logging.warning(f"Could not parse date in field '{field}': {e}")
                
                combined_text = " ".join(text_parts)
                return self.preprocess_text(combined_text, dataset_type)  # Pass dataset_type here
            
            logging.warning(f"Unsupported record type: {type(record)}")
            return ""
            
        except Exception as e:
            logging.error(f"Error in text extraction: {str(e)}")
            return ""

    def process_with_memory_management(self, texts: Generator[str, None, None], dataset_name: str) -> Generator[str, None, None]:
        """Memory-efficient text processing using generators"""
        buffer_size = 1000
        buffer = []
        try:
            for text in texts:
                processed_text = self.preprocess_text(text, dataset_name)
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
        available_memory = psutil.virtual_memory().available
        base_chunk_size = 1000  # Minimum chunk size
        
        # Use 5% of available memory, assuming 8KB per text
        memory_based_size = max(base_chunk_size, int(available_memory * 0.05 / 8192))
        
        # Cap at a reasonable maximum
        max_chunk_size = 100000
        return min(memory_based_size, max_chunk_size)

    def _get_optimal_workers(self) -> int:
        """Dynamically adjust worker count based on system resources"""
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            self.current_workers = self.min_workers
        elif memory.percent > 80:
            self.current_workers = max(self.min_workers, self.current_workers // 2)
        elif memory.percent < 60 and self.current_workers < multiprocessing.cpu_count():
            self.current_workers = min(self.current_workers * 2, multiprocessing.cpu_count())
            
        return self.current_workers
        
    def get_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory"""
        available_memory = psutil.virtual_memory().available
        return max(1000, int(available_memory * 0.1 / 8192))  # 8KB per text estimate

    def _process_chunk(self, chunk: List[str], worker_id: int, dataset_name: str) -> List[str]:
        """Process a chunk of texts with improved memory management"""
        processed = []
        try:
            # Process in smaller batches with progress bar
            for i in range(0, len(chunk), self.batch_size):
                batch = chunk[i:i + self.batch_size]
                
                # Process each text in the batch
                for text in batch:
                    processed_text = self.preprocess_text(text, dataset_name)
                    if processed_text:  # Only add non-empty processed texts
                        processed.append(processed_text)
                        
                # Memory management
                if psutil.virtual_memory().percent > self.memory_threshold * 100:
                    gc.collect()
                    
        except Exception as e:
            logging.error(f"Worker {worker_id}: Chunk processing error: {str(e)}")
            raise
            
        return processed

    def _process_generic_texts(
        self,
        texts: List[str], 
        output_path: Path, 
        chunk_size: int,
        num_workers: int,
        dataset_name: str = 'general'  # Add dataset_name parameter with default
    ) -> Optional[str]:
        """Process texts with improved memory management."""
        try:
            # Calculate memory-safe chunk size
            available_memory = psutil.virtual_memory().available
            optimal_chunk_size = min(chunk_size, max(1000, int(available_memory * 0.05 / 8192)))
            max_workers = min(num_workers, optimal_chunk_size // 100)  # Ensure enough data per worker
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i in range(0, len(texts), optimal_chunk_size):
                    # Check memory pressure
                    if psutil.virtual_memory().percent > 85:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        time.sleep(1)
                        optimal_chunk_size = max(1000, optimal_chunk_size // 2)
                        max_workers = max(1, max_workers // 2)
                    
                    chunk = texts[i:i + optimal_chunk_size]
                    worker_chunk_size = max(100, len(chunk) // max_workers)
                    
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = {}
                        for j in range(0, len(chunk), worker_chunk_size):
                            sub_chunk = chunk[j:j + worker_chunk_size]
                            futures[executor.submit(
                                DatasetProcessor._process_chunk_with_retry,
                                sub_chunk,
                                {'max_attempts': 3, 'delay': 1, 'max_delay': 10, 'backoff': 2},
                                dataset_name  # Add dataset_name here
                            )] = j
                        
                        # Process results as they complete
                        processed_chunk = []
                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=30)
                                if result:
                                    processed_chunk.extend(result)
                                    f.write('\n'.join(result) + '\n')
                                    f.flush()
                            except Exception as e:
                                logging.error(f"Chunk {futures[future]} failed: {str(e)}")
                        
                        # Clear memory explicitly
                        del chunk
                        del processed_chunk
                        del futures
                        gc.collect()
            
            return str(output_path)
        
        except Exception as e:
            logging.error(f"Error in _process_generic_texts: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            return None

    def _process_chunk_with_retry(
        self, 
        chunk: List[str], 
        retry_config: Dict[str, Any],
        dataset_name: str = 'general'  # Add dataset_name parameter with default
    ) -> List[str]:
        """Process a chunk of texts with enhanced retry mechanism and adaptive sizing."""
        attempt = 0
        exceptions = []
        current_chunk = chunk
        
        # Calculate initial chunk memory footprint
        chunk_size = sum(len(text.encode('utf-8')) for text in chunk)
        max_chunk_size = self._get_max_chunk_memory()
        
        while attempt < retry_config['max_attempts']:
            try:
                # If chunk is too large, split it
                if chunk_size > max_chunk_size:
                    logging.warning(f"Chunk size ({chunk_size} bytes) exceeds limit ({max_chunk_size} bytes). Splitting...")
                    split_size = max(1, len(current_chunk) // 2)
                    results = []
                    for sub_chunk in [current_chunk[:split_size], current_chunk[split_size:]]:
                        sub_result = self._process_chunk_with_retry(
                            sub_chunk,
                            {**retry_config, 'max_attempts': retry_config['max_attempts'] - attempt},
                            dataset_name  # Pass dataset_name here
                        )
                        results.extend(sub_result)
                    return results
                    
                return self._process_chunk(current_chunk, attempt, dataset_name)  # Pass dataset_name here
                
            except MemoryError as e:
                attempt += 1
                exceptions.append((attempt, str(e), traceback.format_exc()))
                # Reduce chunk size on memory error
                current_chunk = current_chunk[:len(current_chunk)//2]
                chunk_size = sum(len(text.encode('utf-8')) for text in current_chunk)
                
            except Exception as e:
                attempt += 1
                exceptions.append((attempt, str(e), traceback.format_exc()))
                
            if attempt < retry_config['max_attempts']:
                delay = min(
                    retry_config['delay'] * (retry_config['backoff'] ** (attempt - 1)),
                    retry_config['max_delay']
                )
                logging.warning(f"Attempt {attempt} failed. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                self._log_retry_failures(exceptions, current_chunk)
                return []

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available system resources."""
        try:
            # Get available system memory
            memory = psutil.virtual_memory()
            available_memory = memory.available
            
            # Base batch size calculation (assuming ~1KB per text item)
            base_size = max(100, available_memory // (1024 * 1024))  # Convert to MB
            
            # Adjust based on GPU if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_used = torch.cuda.memory_allocated()
                gpu_factor = (gpu_memory - gpu_memory_used) / gpu_memory
                base_size = int(base_size * gpu_factor)
            
            # Adjust based on CPU cores
            cpu_count = multiprocessing.cpu_count()
            base_size = min(base_size, cpu_count * 1000)
            
            # Set reasonable bounds
            min_batch_size = 100
            max_batch_size = 10000
            
            return max(min_batch_size, min(base_size, max_batch_size))
            
        except Exception as e:
            logging.warning(f"Error calculating optimal batch size: {e}")
            return 1000  # Default fallback value

    def _calculate_total_files(self) -> int:
        """Calculate total number of files to be processed across all datasets."""
        total = 0
        try:
            for dataset_config in self.datasets:
                dataset_name = dataset_config.get('name', '')
                if dataset_name == 'openwebtext':
                    # For OpenWebText, estimate based on typical dataset size
                    total += 8000000  # Approximate number of documents in OpenWebText
                elif dataset_name == 'wikipedia':
                    # For Wikipedia, estimate based on typical dump size
                    total += 6000000  # Approximate number of articles
                else:
                    # For custom datasets, check the actual file count
                    data_path = Path(dataset_config.get('path', ''))
                    if data_path.exists() and data_path.is_dir():
                        total += sum(1 for _ in data_path.rglob('*') 
                                   if self.file_validator.validate_file(_)[0])
            
            return max(1, total)  # Ensure at least 1 to avoid division by zero
            
        except Exception as e:
            logging.error(f"Error calculating total files: {str(e)}")
            return 1000  # Return a default value if calculation fails

    def process(self) -> List[str]:
        """
        Process all datasets and return list of processed file paths.
        """
        processed_files = []
        try:
            for dataset_config in self.datasets:
                dataset_name = dataset_config.get('name', '')
                logging.info(f"Processing dataset: {dataset_name}")
                
                try:
                    if dataset_name == 'openwebtext':
                        files = self._process_openwebtext(dataset_config)
                    elif dataset_name == 'wikipedia':
                        files = self._process_wikipedia(dataset_config)
                    else:
                        files = self._process_custom_dataset(dataset_config)
                    
                    if files:
                        processed_files.extend(files)
                        logging.info(f"Successfully processed {len(files)} files from {dataset_name}")
                    else:
                        logging.warning(f"No files processed from dataset: {dataset_name}")
                        
                except Exception as e:
                    logging.error(f"Error processing dataset {dataset_name}: {str(e)}")
                    continue
            
            if not processed_files:
                raise ValueError("No files were successfully processed")
                
            return processed_files
            
        except Exception as e:
            logging.error(f"Error in dataset processing: {str(e)}")
            raise

    def _process_openwebtext(self, config: Dict[str, Any]) -> List[str]:
        """Process OpenWebText dataset with enhanced error handling and progress tracking."""
        try:
            logging.info("Starting OpenWebText dataset processing...")
            
            # Configure dataset loading with proper error handling
            download_config = config.get('download_config', {})
            cache_dir = download_config.get('cache_dir', '.cache')
            
            # Ensure cache directory exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Load dataset with progress tracking
            logging.info("Loading OpenWebText dataset...")
            try:
                dataset = load_dataset(
                    'openwebtext',
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    streaming=True  # Use streaming for memory efficiency
                )
            except Exception as e:
                logging.error(f"Failed to load OpenWebText dataset: {str(e)}")
                # Try alternative loading method
                try:
                    dataset = load_dataset(
                        'openwebtext',
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        streaming=False
                    )
                except Exception as e2:
                    logging.error(f"All attempts to load dataset failed: {str(e2)}")
                    raise
            
            # Setup output directory
            output_dir = Path(self.config.local_data_path) / "processed" / "openwebtext"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            processed_files = []
            batch_size = 10000  # Process in smaller batches
            
            # Process each split in the dataset
            for split_name, split_dataset in dataset.items():
                logging.info(f"Processing split: {split_name}")
                
                # Process in batches with progress tracking
                current_batch = []
                current_file_idx = 0
                
                with tqdm(desc=f"Processing {split_name}", unit="texts") as pbar:
                    for item in split_dataset:
                        if 'text' in item and item['text']:
                            current_batch.append(item['text'])
                            
                            if len(current_batch) >= batch_size:
                                # Process current batch
                                output_path = output_dir / f"{split_name}_batch_{current_file_idx}.txt"
                                result = self._process_generic_texts(
                                    texts=current_batch,
                                    output_path=output_path,
                                    chunk_size=self.batch_size,
                                    num_workers=self.current_workers,
                                    dataset_name='openwebtext'
                                )
                                
                                if result:
                                    processed_files.append(result)
                                    current_file_idx += 1
                                
                                current_batch = []
                                pbar.update(batch_size)
                
                    # Process remaining texts
                    if current_batch:
                        output_path = output_dir / f"{split_name}_batch_{current_file_idx}.txt"
                        result = self._process_generic_texts(
                            texts=current_batch,
                            output_path=output_path,
                            chunk_size=self.batch_size,
                            num_workers=self.current_workers,
                            dataset_name='openwebtext'
                        )
                        
                        if result:
                            processed_files.append(result)
            
            if not processed_files:
                logging.warning("No files were processed from OpenWebText dataset")
                return []
            
            logging.info(f"Successfully processed {len(processed_files)} files from OpenWebText")
            return processed_files
            
        except Exception as e:
            logging.error(f"Error processing OpenWebText: {str(e)}")
            logging.error(traceback.format_exc())
            return []

    def _process_wikipedia(self, config: Dict[str, Any]) -> List[str]:
        """Process Wikipedia dataset."""
        try:
            # Load dataset
            dataset = load_dataset(
                'wikipedia',
                '20220301.en',
                cache_dir=config.get('download_config', {}).get('cache_dir'),
                trust_remote_code=config.get('download_config', {}).get('trust_remote_code', True)
            )
            
            # Process texts
            output_dir = Path(self.config.local_data_path) / "processed" / "wikipedia"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            processed_files = []
            for split in dataset:
                output_path = output_dir / f"{split}_processed.txt"
                texts = [text['text'] for text in dataset[split]]
                
                result = self._process_generic_texts(
                    texts=texts,
                    output_path=output_path,
                    chunk_size=self.batch_size,
                    num_workers=self.current_workers,
                    dataset_name='wikipedia'
                )
                
                if result:
                    processed_files.append(result)
            
            return processed_files
            
        except Exception as e:
            logging.error(f"Error processing Wikipedia: {str(e)}")
            return []

    def _process_custom_dataset(self, config: Dict[str, Any]) -> List[str]:
        """Process custom dataset from local files."""
        try:
            data_path = Path(config.get('path', ''))
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {data_path}")
            
            output_dir = Path(self.config.local_data_path) / "processed" / data_path.name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            processed_files = []
            valid_files = [
                f for f in data_path.rglob('*')
                if self.file_validator.validate_file(f)[0]
            ]
            
            for file_path in valid_files:
                relative_path = file_path.relative_to(data_path)
                output_path = output_dir / f"{relative_path.stem}_processed.txt"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = f.readlines()
                
                result = self._process_generic_texts(
                    texts=texts,
                    output_path=output_path,
                    chunk_size=self.batch_size,
                    num_workers=self.current_workers,
                    dataset_name='custom'
                )
                
                if result:
                    processed_files.append(result)
            
            return processed_files
            
        except Exception as e:
            logging.error(f"Error processing custom dataset: {str(e)}")
            return []

class GPUMemoryMonitor:
    """Enhanced GPU memory monitor with fallback mechanisms"""
    
    def __init__(self, initial_threshold: float = 0.8):
        self.threshold = initial_threshold
        self.adjustment_factor = 0.9
        self.min_threshold = 0.5
        self.history: List[float] = []
        self._nvidia_smi_available = self._check_nvidia_smi()
        
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except:
            return False
            
    def _get_gpu_memory_info(self) -> Tuple[int, int]:
        """Get GPU memory info with fallback mechanisms"""
        try:
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                max_memory = torch.cuda.max_memory_allocated()
                
                # If max_memory is 0, try nvidia-smi
                if max_memory == 0 and self._nvidia_smi_available:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    return info.used, info.total
                    
                return current_memory, max_memory or current_memory
                
        except Exception as e:
            logging.warning(f"Failed to get GPU memory info: {e}")
            
        return 0, 0
        
    def should_pause(self) -> bool:
        """Check if processing should pause based on memory usage"""
        if not torch.cuda.is_available():
            return False
            
        current_memory, max_memory = self._get_gpu_memory_info()
        if max_memory == 0:
            return False
            
        usage_ratio = current_memory / max_memory
        self.update_threshold(usage_ratio)
        
        return usage_ratio > self.threshold

class MemoryMonitor:
    """Monitor system memory usage"""
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        
    def should_pause(self) -> bool:
        memory = psutil.virtual_memory()
        return memory.percent > (self.threshold * 100)

class AsyncProcessPool:
    """Asynchronous process pool with resource management"""
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.pool = None
        
    async def __aenter__(self):
        self.pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.shutdown(wait=True)
            
    def submit(self, fn, *args, **kwargs):
        return asyncio.wrap_future(self.pool.submit(fn, *args, **kwargs))

class AsyncIterator:
    """Async iterator for concurrent task processing"""
    def __init__(self, tasks):
        self.tasks = tasks
        self.pending = set(tasks)
        
    async def __aiter__(self):
        return self
        
    async def __anext__(self):
        if not self.pending:
            raise StopAsyncIteration
        
        done, self.pending = await asyncio.wait(
            self.pending, 
            return_when=asyncio.FIRST_COMPLETED
        )
        return done.pop()

class SynchronizedProgress:
    """Thread-safe progress bar with enhanced error handling"""
    
    def __init__(self, total: int, desc: str = None):
        self.total = total
        self.desc = desc
        self.current = 0
        self._lock = threading.Lock()
        self._error_count = 0
        self._max_errors = 3
        self._closed = False
        self._last_update = 0
        self._update_interval = 0.1  # seconds
        self.pbar = tqdm(total=total, desc=desc)
        
    def update(self, n: int = 1):
        """Thread-safe progress update with error recovery"""
        if self._closed:
            return
            
        try:
            with self._lock:
                current_time = time.time()
                if current_time - self._last_update >= self._update_interval:
                    self.current += n
                    # Ensure we don't exceed total
                    self.current = min(self.current, self.total)
                    # Update progress bar
                    try:
                        self.pbar.n = self.current
                        self.pbar.refresh()
                        self._last_update = current_time
                    except Exception as e:
                        self._handle_update_error(e)
                        
        except Exception as e:
            self._handle_update_error(e)
            
    def _handle_update_error(self, error: Exception):
        """Handle progress bar update errors"""
        self._error_count += 1
        logging.warning(f"Progress update error ({self._error_count}/{self._max_errors}): {str(error)}")
        
        if self._error_count >= self._max_errors:
            logging.error("Too many progress bar errors, switching to basic logging")
            self._closed = True
            try:
                self.pbar.close()
            except:
                pass
            # Log final progress
            logging.info(f"Progress: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)")

class TokenizerPathManager:
    """Manages tokenizer save paths with backup and restore capabilities"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.backup_dir = self.base_path / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def get_latest_backup(self) -> Optional[Path]:
        """Get the most recent backup file"""
        backups = sorted(self.backup_dir.glob("*_backup_*.json"), 
                        key=lambda x: x.stat().st_mtime,
                        reverse=True)
        return backups[0] if backups else None
        
    def restore_from_backup(self, tokenizer_path: Path, specific_backup: Optional[Path] = None) -> bool:
        """Restore tokenizer from backup"""
        try:
            backup_path = specific_backup or self.get_latest_backup()
            if not backup_path or not backup_path.exists():
                logging.error("No backup file found")
                return False
                
            shutil.copy2(backup_path, tokenizer_path)
            logging.info(f"Restored tokenizer from backup: {backup_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to restore from backup: {str(e)}")
            return False

class ChunkManager:
    """Manages data chunking with dynamic worker adjustment"""
    
    def __init__(self, initial_workers: int = multiprocessing.cpu_count()):
        self.current_workers = initial_workers
        self.min_workers = 1
        self.memory_monitor = MemoryMonitor()
        
    def adjust_workers(self) -> int:
        """Dynamically adjust worker count based on system resources"""
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            self.current_workers = self.min_workers
        elif memory.percent > 80:
            self.current_workers = max(self.min_workers, self.current_workers // 2)
        elif memory.percent < 60 and self.current_workers < multiprocessing.cpu_count():
            self.current_workers = min(self.current_workers * 2, multiprocessing.cpu_count())
            
        return self.current_workers
        
    def get_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory"""
        available_memory = psutil.virtual_memory().available
        return max(1000, int(available_memory * 0.1 / 8192))  # 8KB per text estimate

class AsyncFileProcessor:
    """Enhanced asynchronous file operations handler"""
    
    def __init__(self, max_buffer_size: int = 10 * 1024 * 1024):  # 10MB default buffer
        self.max_buffer_size = max_buffer_size
        self.buffer = []
        self.buffer_size = 0
        self.file_locks: Dict[Path, asyncio.Lock] = {}

    async def process_file(self, file_path: Path, operation: str, data: Any = None) -> Optional[Any]:
        """Process file operations with automatic retry and logging"""
        file_id = file_path.stem[:8]  # Use first 8 chars of filename as ID
        
        for attempt in range(3):  # Max 3 retries
            try:
                if operation == 'read':
                    return await self._read_file(file_path)
                elif operation == 'write':
                    await self._write_file(file_path, data)
                elif operation == 'append':
                    await self._append_to_file(file_path, data)
                break
            except Exception as e:
                logging.error(f"File operation failed [ID: {file_id}] (attempt {attempt + 1}): {str(e)}")
                if attempt == 2:  # Last attempt
                    raise

    async def _read_file(self, file_path: Path) -> str:
        """Read file with proper encoding detection"""
        async with aiofiles.open(file_path, mode='rb') as f:
            raw_data = await f.read()
            
        # Detect encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
        
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            logging.warning(f"Fallback to utf-8 with error handling for {file_path}")
            return raw_data.decode('utf-8', errors='replace')

    async def _write_file(self, file_path: Path, data: str) -> None:
        """Write to file with locking"""
        if file_path not in self.file_locks:
            self.file_locks[file_path] = asyncio.Lock()
            
        async with self.file_locks[file_path]:
            async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
                await f.write(data)

    async def _append_to_file(self, file_path: Path, data: str) -> None:
        """Append to file with buffering"""
        self.buffer.append(data)
        self.buffer_size += len(data.encode('utf-8'))
        
        if self.buffer_size >= self.max_buffer_size:
            await self._flush_buffer(file_path)

    async def _flush_buffer(self, file_path: Path) -> None:
        """Flush buffer to file"""
        if not self.buffer:
            return
            
        if file_path not in self.file_locks:
            self.file_locks[file_path] = asyncio.Lock()
            
        async with self.file_locks[file_path]:
            async with aiofiles.open(file_path, mode='a', encoding='utf-8') as f:
                await f.write(''.join(self.buffer))
                
        self.buffer = []
        self.buffer_size = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure all buffers are flushed on exit"""
        for file_path in self.file_locks:
            await self._flush_buffer(file_path)

class DatasetConfigManager:
    """Enhanced dataset configuration manager with custom preprocessing pipelines"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("dataset_configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.last_reload: Dict[str, float] = {}
        self.reload_interval = 300  # 5 minutes
        
        # Register custom preprocessing functions
        self.preprocessing_registry = {
            'medical': self._medical_preprocessing,
            'scientific': self._scientific_preprocessing,
            'general': self._general_preprocessing
        }

    def get_preprocessing_pipeline(self, dataset_name: str) -> Callable:
        """Get dataset-specific preprocessing pipeline"""
        config = self.get_config(dataset_name)
        pipeline_name = config.get('preprocessing', {}).get('pipeline', 'general')
        return self.preprocessing_registry.get(pipeline_name, self._general_preprocessing)

    def _medical_preprocessing(self, text: str) -> str:
        """Medical domain-specific preprocessing"""
        # Standardize medical abbreviations
        medical_abbreviations = {
            'pt': 'patient',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'hx': 'history'
        }
        
        for abbr, full in medical_abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', full, text, flags=re.IGNORECASE)
            
        # Remove PHI patterns
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{2}/\d{2}/\d{4}\b',   # Dates
            r'\b[A-Z]{2}\d{6}\b'        # Medical record numbers
        ]
        
        for pattern in phi_patterns:
            text = re.sub(pattern, '[REDACTED]', text)
            
        return text

    def _scientific_preprocessing(self, text: str) -> str:
        """Scientific text preprocessing"""
        # Standardize units
        unit_patterns = {
            r'\bmg/dl\b': 'mg/dL',
            r'\bug/ml\b': 'μg/mL',
            r'\bng/ml\b': 'ng/mL'
        }
        
        for pattern, replacement in unit_patterns.items():
            text = re.sub(pattern, replacement, text)
            
        # Handle mathematical expressions
        text = re.sub(r'(\d+)\s*\^\s*(\d+)', r'\1^\2', text)  # Fix spacing in exponents
        
        return text

    def _general_preprocessing(self, text: str) -> str:
        """General purpose preprocessing"""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove special characters
        return text.strip()

class EnhancedLogger:
    """Enhanced logging with context tracking and structured output"""
    
    def __init__(self, log_file: Path, max_file_size: int = 10 * 1024 * 1024):
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.context: Dict[str, Any] = {}
        
        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging with rotation and formatting"""
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(context)s] %(message)s'
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('tokenizer')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def set_context(self, **kwargs):
        """Set context for logging"""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear current context"""
        self.context.clear()

    def _format_context(self) -> str:
        """Format context for log message"""
        return ' '.join(f'{k}={v}' for k, v in self.context.items())

    def info(self, message: str, **kwargs):
        """Log info message with context"""
        extra = {'context': self._format_context()}
        extra.update(kwargs)
        self.logger.info(message, extra=extra)

    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error message with context and optional stack trace"""
        extra = {'context': self._format_context()}
        extra.update(kwargs)
        self.logger.error(message, exc_info=exc_info, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        extra = {'context': self._format_context()}
        extra.update(kwargs)
        self.logger.warning(message, extra=extra)

    @contextmanager
    def context_scope(self, **kwargs):
        """Context manager for temporary context"""
        previous = self.context.copy()
        self.set_context(**kwargs)
        try:
            yield
        finally:
            self.context = previous


###############################################################################
# Main Execution
###############################################################################
def main():
    """Enhanced main function with better configuration and error handling."""
    try:
        # Remove any existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        parser = argparse.ArgumentParser(description='Medical text tokenizer')
        
        # Enhanced argument parsing
        parser.add_argument('--local_data_path', type=str, required=True,
                          help="Path to store processed data and tokenizer")
        parser.add_argument('--vocab_size', type=int, default=60000,
                          help="Vocabulary size for the tokenizer")
        parser.add_argument('--min_frequency', type=int, default=2,
                          help="Minimum frequency for BPE merges")
        parser.add_argument('--log_file', type=str, default="medical_tokenization.log",
                          help="Log file path")
        parser.add_argument('--cache_dir', type=str, default=".data_cache",
                          help="Cache directory for datasets")
        parser.add_argument('--chunk_size', type=int, default=1000,
                          help="Initial chunk size for processing")
        parser.add_argument('--max_workers', type=int, 
                          default=max(1, multiprocessing.cpu_count() - 1),
                          help="Maximum number of worker processes")
        parser.add_argument('--memory_threshold', type=float, default=0.8,
                          help="Memory usage threshold (0-1) for adaptive processing")
        
        # Parse only known args to avoid conflicts with unittest arguments
        args, unknown = parser.parse_known_args()

        # Setup enhanced logging
        setup_logging(args.log_file)

        # Create necessary directories
        tokens_dir = Path(args.local_data_path) / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Updated dataset configuration with validation
        datasets = [
            {
                "name": "openwebtext",
                "download_config": {
                    "cache_dir": str(cache_dir),
                    "trust_remote_code": True
                }
            }
        ]
        
        # Initialize configuration
        config = Config(
            local_data_path=args.local_data_path,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            log_file=args.log_file,
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
            memory_threshold=args.memory_threshold
        )
        
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        with memory_manager.monitor_memory("dataset_processing"):
            # Process datasets with progress tracking
            dataset_processor = DatasetProcessor(datasets, config)
            processed_files = dataset_processor.process()
            
            if not processed_files:
                logging.error("No files were successfully processed")
                return
            
            # Train tokenizer with memory monitoring
            tokenizer = MedicalTokenizer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency
            )
            
            tokenizer_path = tokens_dir / "Medical_tokenizer.json"
            
            # Create backup of existing tokenizer if it exists
            if tokenizer_path.exists():
                backup_path = tokenizer_path.with_suffix('.backup')
                shutil.copy2(tokenizer_path, backup_path)
                logging.info(f"Created backup at {backup_path}")
            
            # Train with progress tracking
            with tqdm(total=1, desc="Training tokenizer") as pbar:
                tokenizer.train(processed_files, str(tokenizer_path))
                pbar.update(1)
            
            logging.info(f"Successfully trained tokenizer at {tokenizer_path}")
            
            # Validate trained tokenizer
            try:
                test_tokenizer = MedicalTokenizer()
                test_tokenizer.load(str(tokenizer_path))
                logging.info("Tokenizer validation successful")
            except Exception as e:
                logging.error(f"Tokenizer validation failed: {e}")
                if backup_path.exists():
                    shutil.copy2(backup_path, tokenizer_path)
                    logging.info("Restored from backup")
                raise
            
        logging.info("Processing and training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

def _process_generic_texts(
    texts: List[str], 
    output_path: Path, 
    chunk_size: int, 
    num_workers: int,
    dataset_name: str = 'general'  # Add dataset_name parameter with default
) -> Optional[str]:
    """Process texts with improved memory management."""
    try:
        # Calculate memory-safe chunk size
        available_memory = psutil.virtual_memory().available
        optimal_chunk_size = min(chunk_size, max(1000, int(available_memory * 0.05 / 8192)))
        max_workers = min(num_workers, optimal_chunk_size // 100)  # Ensure enough data per worker
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(0, len(texts), optimal_chunk_size):
                # Check memory pressure
                if psutil.virtual_memory().percent > 85:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(1)
                    optimal_chunk_size = max(1000, optimal_chunk_size // 2)
                    max_workers = max(1, max_workers // 2)
                
                chunk = texts[i:i + optimal_chunk_size]
                worker_chunk_size = max(100, len(chunk) // max_workers)
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for j in range(0, len(chunk), worker_chunk_size):
                        sub_chunk = chunk[j:j + worker_chunk_size]
                        futures[executor.submit(
                            DatasetProcessor._process_chunk_with_retry,
                            sub_chunk,
                            {'max_attempts': 3, 'delay': 1, 'max_delay': 10, 'backoff': 2},
                            dataset_name  # Add dataset_name here
                        )] = j
                    
                    # Process results as they complete
                    processed_chunk = []
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=30)
                            if result:
                                processed_chunk.extend(result)
                                f.write('\n'.join(result) + '\n')
                                f.flush()
                        except Exception as e:
                            logging.error(f"Chunk {futures[future]} failed: {str(e)}")
                
                # Clear memory explicitly
                del chunk
                del processed_chunk
                del futures
                gc.collect()
        
        return str(output_path)
        
    except Exception as e:
        logging.error(f"Error in _process_generic_texts: {str(e)}")
        if output_path.exists():
            output_path.unlink()
        return None

def dynamic_chunk_size():
    total_memory = psutil.virtual_memory().total
    return max(1000, int(total_memory * 0.1 / 8192))  # Assuming 8KB per text

chunk_size = dynamic_chunk_size()

class TestUtils(unittest.TestCase):
    def test_with_retry(self):
        # Add test cases for with_retry
        #To be added later. For all 3. 
        pass

    def test_managed_temp_file(self):
        # Add test cases for managed_temp_file
        pass

    def test_split_chunk(self):
        # Add test cases for split_chunk
        pass

if __name__ == '__main__':
    unittest.main()

def split_chunk(chunk: List[Any], num_parts: int) -> Generator[List[Any], None, None]:
    """
    Split a chunk into approximately equal parts with proper validation.
    
    Args:
        chunk: List to be split
        num_parts: Number of parts to split into
        
    Yields:
        Generator of sub-chunks
    """
    if not chunk:
        return
    
    # Ensure num_parts is reasonable
    num_parts = min(num_parts, len(chunk))
    
    # Calculate base size and remainder
    base_size = len(chunk) // num_parts
    remainder = len(chunk) % num_parts
    
    start = 0
    for i in range(num_parts):
        # Add one extra item to some chunks if there's remainder
        end = start + base_size + (1 if i < remainder else 0)
        if start < len(chunk):
            yield chunk[start:end]
        start = end


