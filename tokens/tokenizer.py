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
from datasets import load_dataset, DatasetDict, IterableDataset
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
import numpy as np

###############################################################################
# Configuration
###############################################################################
class Config:
    """Configuration class for tokenizer training."""
    def __init__(
        self,
        local_data_path: str,
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        log_file: str = "tokenizer.log",
        chunk_size: int = 1000,
        max_workers: Optional[int] = None,
        memory_threshold: float = 0.8,
        cache_dir: str = ".cache",
        allowed_extensions: Set[str] = {'.txt', '.json', '.jsonl', '.csv'},
        allowed_mimetypes: Set[str] = {'text/plain', 'application/json', 'text/csv'},
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        gpu_memory_threshold: float = 0.8
    ):
        self.local_data_path = local_data_path
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.log_file = log_file
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.memory_threshold = memory_threshold
        self.cache_dir = cache_dir
        self.allowed_extensions = allowed_extensions
        self.allowed_mimetypes = allowed_mimetypes
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
    """Enhanced memory manager with alerts and dynamic thresholds"""
    def __init__(self, alert_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 minutes between alerts
        
    def check_memory(self) -> Tuple[bool, str]:
        """Check memory status with detailed reporting"""
        memory = psutil.virtual_memory()
        current_usage = memory.percent / 100
        
        status_message = f"Memory usage: {memory.percent}%"
        
        if current_usage >= self.critical_threshold:
            if time.time() - self.last_alert_time > self.alert_cooldown:
                self.last_alert_time = time.time()
                logging.critical(f"Critical memory usage: {memory.percent}%")
            return True, f"CRITICAL: {status_message}"
            
        elif current_usage >= self.alert_threshold:
            if time.time() - self.last_alert_time > self.alert_cooldown:
                self.last_alert_time = time.time()
                logging.warning(f"High memory usage: {memory.percent}%")
            return True, f"WARNING: {status_message}"
            
        return False, status_message

    def get_safe_chunk_size(self, item_size_bytes: int = 8192) -> int:
        """Calculate safe chunk size based on available memory"""
        available_memory = psutil.virtual_memory().available
        target_memory = available_memory * 0.5  # Use 50% of available memory
        return max(1000, int(target_memory / item_size_bytes))


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
    """Enhanced tokenizer class with dynamic vocabulary sizing and comprehensive preprocessing."""
    
    def __init__(
        self,
        vocab_size: Optional[int] = 60000,
        min_frequency: int = 2,
        padding_strategy: str = 'longest',
        truncation_strategy: str = 'longest_first',
        max_length: int = 512,
        normalize: bool = True
    ):
        # Initialize tokenizer with BPE model and special tokens
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        # Set up pre-tokenizer first
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        
        # Add special tokens to the tokenizer's model
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
        for token in special_tokens:
            _ = self.tokenizer.token_to_id(token)
        
        # Configure padding and truncation
        self.padding_config = {
            'strategy': padding_strategy,
            'pad_token': "[PAD]",
            'pad_token_id': 0,
            'pad_to_multiple_of': 8
        }
        
        self.truncation_config = {
            'strategy': truncation_strategy,
            'max_length': max_length,
            'stride': 0,
            'direction': 'right'
        }
        
        # Special tokens configuration
        self.special_tokens = {
            'pad_token': "[PAD]",
            'unk_token': "[UNK]",
            'cls_token': "[CLS]",
            'sep_token': "[SEP]",
            'mask_token': "[MASK]",
            'bos_token': "[BOS]",
            'eos_token': "[EOS]"
        }
        
        # Normalization configuration
        self.normalize = normalize
        if normalize:
            self.normalizer = self._setup_normalizer()
            self.tokenizer.normalizer = self.normalizer
        
        # Dynamic vocabulary sizing
        self.initial_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab_size = self._calculate_dynamic_vocab_size()
        
        # Configure tokenizer components
        self._setup_tokenizer()

    def _setup_normalizer(self):
        """Setup text normalization rules."""
        from tokenizers import normalizers
        return normalizers.Sequence([
            normalizers.NFKC(),  # Unicode normalization
            normalizers.Replace(r'[\n\r\t]+', ' '),  # Replace newlines/tabs
            normalizers.Replace(r'\s+', ' '),  # Normalize whitespace
            normalizers.Replace(r'(?<=\d)[,.](?=\d{3})', ''),  # Handle numbers
            normalizers.Lowercase(),  # Convert to lowercase
            # Medical-specific normalizations
            normalizers.Replace(r'\b(?:mg/dl|mg/dL)\b', 'mg/dL'),  # Standardize units
            normalizers.Replace(r'\b(?:mcg|µg|ug)/(?:ml|mL)\b', 'μg/mL'),
            normalizers.Replace(r'\b(?:ng)/(?:ml|mL)\b', 'ng/mL')
        ])

    def _setup_tokenizer(self):
        """Configure tokenizer components."""
        # Set pre-tokenizer
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        
        # Set normalizer if enabled
        if self.normalize:
            self.tokenizer.normalizer = self.normalizer
        
        # Configure post-processor for special token handling
        from tokenizers.processors import TemplateProcessing
        
        try:
            # Create special tokens as a list of tuples (token, id)
            special_tokens = [
                (self.special_tokens['cls_token'], 0),  # CLS token with ID 0
                (self.special_tokens['sep_token'], 1),  # SEP token with ID 1
            ]
            
            self.tokenizer.post_processor = TemplateProcessing(
                single=f"$A {self.special_tokens['sep_token']}",
                pair=f"{self.special_tokens['cls_token']} $A {self.special_tokens['sep_token']} $B {self.special_tokens['sep_token']}",
                special_tokens=special_tokens  # Pass as sequence of tuples
            )
        except Exception as e:
            logging.error(f"Failed to set up post-processor: {str(e)}")
            # Fallback to simpler configuration if needed
            self.tokenizer.post_processor = TemplateProcessing(
                single="$A",
                pair="$A $B",
                special_tokens=[]  # Empty sequence for fallback
            )

    def _calculate_dynamic_vocab_size(self) -> int:
        """Calculate vocabulary size."""
        return 60000  # Fixed vocabulary size

    def _analyze_dataset_vocabulary(self) -> int:
        """Analyze dataset to count unique tokens."""
        if not hasattr(self.tokenizer, 'pre_tokenizer') or self.tokenizer.pre_tokenizer is None:
            logging.error("Pre-tokenizer not initialized")
            return 0
        
        unique_tokens = set()
        sample_size = 100000  # Limit analysis to first 100K tokens
        
        try:
            # Sample text for analysis
            sample_text = "John was admitted to the emergency room after experiencing severe headaches and blurred vision. The physician ordered a CT scan to rule out intracranial hemorrhage. Meanwhile, artificial intelligence tools in radiology have improved diagnostic accuracy significantly. In other news, OpenAI has released a state-of-the-art language model capable of generating coherent and contextually aware text."
            tokens = self.tokenizer.pre_tokenizer.pre_tokenize_str(sample_text)
            
            for token, _ in tokens:
                unique_tokens.add(token)
                if len(unique_tokens) >= sample_size:
                    break
            
            return len(unique_tokens)
            
        except Exception as e:
            logging.warning(f"Vocabulary analysis failed: {str(e)}")
            return 0

    def train(self, files: List[str], save_path: str):
        """Train tokenizer with dynamic configuration."""
        try:
            # Configure trainer with current settings
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=list(self.special_tokens.values()),
                initial_alphabet=ByteLevel.alphabet(),
                show_progress=True
            )
            
            # Train tokenizer
            self.tokenizer.train(files=files, trainer=trainer)
            
            # Save configuration along with tokenizer
            config = {
                'vocab_size': self.vocab_size,
                'min_frequency': self.min_frequency,
                'padding': self.padding_config,
                'truncation': self.truncation_config,
                'special_tokens': self.special_tokens,
                'normalize': self.normalize
            }
            
            # Save tokenizer and config
            self.tokenizer.save(save_path)
            with open(f"{save_path}.config", 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: bool = True
    ) -> Dict[str, Union[List[int], Tensor]]:
        """Enhanced encoding with configurable padding and truncation."""
        if isinstance(texts, str):
            texts = [texts]
            
        # Apply encoding with current configuration
        encodings = self.tokenizer.encode_batch(texts)
        
        # Handle padding if requested
        if padding:
            max_len = max(len(enc.ids) for enc in encodings)
            if max_length:
                max_len = min(max_len, max_length)
                
            for enc in encodings:
                padding_length = max_len - len(enc.ids)
                if padding_length > 0:
                    enc.ids.extend([self.padding_config['pad_token_id']] * padding_length)
                    enc.attention_mask.extend([0] * padding_length)
                    
        # Handle truncation if requested
        if truncation and max_length:
            for enc in encodings:
                if len(enc.ids) > max_length:
                    enc.ids = enc.ids[:max_length]
                    enc.attention_mask = enc.attention_mask[:max_length]
                    
        # Prepare output
        output = {
            'input_ids': [enc.ids for enc in encodings],
            'attention_mask': [enc.attention_mask for enc in encodings]
        }
        
        # Convert to tensors if requested
        if return_tensors:
            output = {k: torch.tensor(v) for k, v in output.items()}
            
        return output


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

    def validate_file(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to read some content
                content = f.read(1024)  # Read first 1KB
                if not content.strip():
                    return False
                return True
        except UnicodeDecodeError:
            # Try to detect encoding
            with open(file_path, 'rb') as f:
                raw = f.read(1024)
                result = chardet.detect(raw)
                if result['encoding']:
                    try:
                        with open(file_path, 'r', encoding=result['encoding']) as f:
                            content = f.read(1024)
                            return bool(content.strip())
                    except:
                        pass
            return False
        except Exception as e:
            logging.error(f"File validation error for {file_path}: {str(e)}")
            return False

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
        self.batch_size = self._calculate_optimal_batch_size()
        self.current_workers = self._calculate_optimal_workers()
        self.cache_dir = Path(config.cache_dir)
        
        # Create necessary directories
        self.output_dir = Path(config.local_data_path) / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _process_local_dataset(self, config: Dict[str, Any]) -> List[str]:
        """Process local dataset from directory."""
        try:
            # Get path from config
            data_path = Path(config.get('config', {}).get('path', ''))
            if not data_path.exists():
                logging.error(f"Dataset path does not exist: {data_path}")
                return []
            
            if not data_path.is_dir():
                logging.error(f"Dataset path is not a directory: {data_path}")
                return []
            
            # Create output directory for this dataset
            dataset_output_dir = self.output_dir / "local"
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            
            processed_files = []
            pattern = config.get('config', {}).get('pattern', '*.txt')
            
            # Find all text files
            text_files = list(data_path.glob(pattern))
            if not text_files:
                logging.warning(f"No {pattern} files found in {data_path}")
                return []
            
            logging.info(f"Found {len(text_files)} files to process")
            
            # Process each file
            for file_path in text_files:
                try:
                    if not self._is_valid_file(file_path):
                        continue
                        
                    output_path = dataset_output_dir / f"{file_path.stem}_processed.txt"
                    
                    # Read and process file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    if not text.strip():
                        logging.warning(f"Empty file: {file_path}")
                        continue
                    
                    # Write processed text
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    processed_files.append(str(output_path))
                    logging.info(f"Successfully processed: {file_path}")
                    
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {str(e)}")
                    continue
            
            return processed_files
            
        except Exception as e:
            logging.error(f"Error processing local dataset: {str(e)}")
            return []

    def _process_huggingface_dataset(self, config: Dict[str, Any]) -> List[str]:
        """Process HuggingFace dataset."""
        try:
            dataset_name = config.get('config', {}).get('dataset_name')
            if not dataset_name:
                logging.error("No dataset name provided")
                return []
            
            logging.info(f"Loading HuggingFace dataset: {dataset_name}")
            
            # Create output directory for this dataset
            dataset_output_dir = self.output_dir / dataset_name.replace('/', '_')
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load dataset
            try:
                if dataset_name == "openwebtext":
                    # Handle streaming for OpenWebText
                    dataset = load_dataset(
                        dataset_name,
                        streaming=True,
                        split='train'
                    )
                    return self._process_streaming_dataset(dataset, dataset_output_dir)
                else:
                    # Handle regular datasets
                    dataset = load_dataset(
                        dataset_name,
                        split=config.get('config', {}).get('split', 'train')
                    )
                    return self._process_regular_dataset(dataset, dataset_output_dir)
                    
            except Exception as e:
                logging.error(f"Error loading dataset {dataset_name}: {str(e)}")
                return []
                
        except Exception as e:
            logging.error(f"Error processing HuggingFace dataset: {str(e)}")
            return []

    def _process_streaming_dataset(self, dataset: IterableDataset, output_dir: Path) -> List[str]:
        """Process streaming dataset with chunking."""
        try:
            output_path = output_dir / "processed.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                batch = []
                
                pbar = tqdm(
                    dataset, 
                    desc="Processing streaming dataset",
                    unit=" samples"
                )
                
                for item in pbar:
                    text = item.get('text', '').strip()
                    if text:
                        batch.append(text)
                        if len(batch) >= self.batch_size:  # Use self.batch_size instead of chunk_size
                            f.write('\n'.join(batch) + '\n')
                            batch = []
                            
                            # Memory management
                            if psutil.virtual_memory().percent > 85:
                                gc.collect()
                                
                if batch:  # Write remaining items
                    f.write('\n'.join(batch) + '\n')
                    
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Error processing streaming dataset: {str(e)}")
            return None

    def _process_regular_dataset(self, dataset: DatasetDict, output_dir: Path) -> List[str]:
        """Process regular (non-streaming) dataset."""
        processed_files = []
        
        try:
            # Get text field name (assuming it's either 'text' or 'content')
            text_field = 'text' if 'text' in dataset.features else 'content'
            
            # Process in batches
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                texts = [item[text_field] for item in batch if item[text_field]]
                
                if texts:
                    output_path = output_dir / f"batch_{i//self.batch_size}.txt"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(texts))
                    processed_files.append(str(output_path))
            
            return processed_files
            
        except Exception as e:
            logging.error(f"Error processing regular dataset: {str(e)}")
            return []

    def process(self) -> List[str]:
        """Process all datasets and return list of processed file paths."""
        all_processed_files = []
        
        try:
            for dataset_config in self.datasets:
                if not isinstance(dataset_config, dict):
                    logging.warning(f"Invalid dataset configuration format: {dataset_config}")
                    continue
                
                dataset_name = dataset_config.get('name', '')
                dataset_type = dataset_config.get('type', '')
                
                logging.info(f"Processing dataset: {dataset_name}")
                
                if dataset_type == 'local':
                    processed_files = self._process_local_dataset(dataset_config)
                elif dataset_type == 'huggingface':
                    processed_files = self._process_huggingface_dataset(dataset_config)
                else:
                    logging.warning(f"Unsupported dataset type: {dataset_type}")
                    continue
                    
                if processed_files:
                    all_processed_files.extend(processed_files)
                else:
                    logging.warning(f"No files processed from dataset: {dataset_name}")
            
            if not all_processed_files:
                raise ValueError("No files were successfully processed")
            
            return all_processed_files
            
        except Exception as e:
            logging.error(f"Error in dataset processing: {str(e)}")
            raise

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
        
    def validate_save_path(self, save_path: Union[str, Path]) -> Path:
        """Validate and prepare save path for tokenizer"""
        save_path = Path(save_path)
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate file extension
        if save_path.suffix.lower() != '.json':
            save_path = save_path.with_suffix('.json')
            
        # Create backup if file exists
        if save_path.exists():
            self._create_backup(save_path)
            
        return save_path
        
    def safe_save(self, tokenizer: Any, save_path: Union[str, Path]) -> None:
        """Safely save tokenizer with backup handling"""
        save_path = self.validate_save_path(save_path)
        temp_path = save_path.with_suffix('.tmp')
        
        try:
            # Save to temporary file first
            tokenizer.save(str(temp_path))
            
            # Rename temporary file to final path
            temp_path.replace(save_path)
            logging.info(f"Successfully saved tokenizer to {save_path}")
            
        except Exception as e:
            logging.error(f"Failed to save tokenizer: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            # Attempt to restore from backup
            self._restore_from_backup(save_path)
            raise
            
    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of existing file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        
        try:
            shutil.copy2(file_path, backup_path)
            logging.info(f"Created backup at {backup_path}")
            return backup_path
        except Exception as e:
            logging.error(f"Failed to create backup: {str(e)}")
            raise
            
    def _restore_from_backup(self, target_path: Path) -> bool:
        """Restore from most recent backup"""
        try:
            latest_backup = self.get_latest_backup()
            if latest_backup and latest_backup.exists():
                shutil.copy2(latest_backup, target_path)
                logging.info(f"Restored from backup: {latest_backup}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to restore from backup: {str(e)}")
            return False
            
    def get_latest_backup(self) -> Optional[Path]:
        """Get the most recent backup file"""
        backups = sorted(self.backup_dir.glob("*_backup_*.json"), 
                        key=lambda x: x.stat().st_mtime,
                        reverse=True)
        return backups[0] if backups else None

class ChunkManager:
    """Manages data chunking with dynamic worker adjustment"""
    
    def __init__(self, initial_workers: int = multiprocessing.cpu_count()):
        self.current_workers = initial_workers
        self.min_workers = 1
        self.memory_monitor = MemoryManager()
        
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
def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """Validate dataset configuration."""
    if not isinstance(config, dict) or 'datasets' not in config:
        logging.error("Invalid configuration: missing 'datasets' key")
        return False

    for dataset in config['datasets']:
        if not isinstance(dataset, dict):
            logging.error(f"Invalid dataset configuration: {dataset}")
            return False
            
        required_fields = {'name', 'type', 'config'}
        missing = required_fields - set(dataset.keys())
        if missing:
            logging.error(f"Dataset missing required fields: {missing}")
            return False
            
        if dataset['type'] not in {'huggingface', 'local'}:
            logging.error(f"Unsupported dataset type: {dataset['type']}")
            return False
            
        if dataset['type'] == 'local':
            path = Path(dataset['config']['path'])
            if not path.exists():
                logging.error(f"Local dataset path does not exist: {path}")
                return False

    return True

def process_local_dataset(input_path: Union[str, Path], output_path: Path, chunk_size: int) -> Optional[str]:
    """Process local dataset files with chunking and progress bar."""
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            logging.error(f"Input path does not exist: {input_path}")
            return None

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If input_path is a directory, process all text files in it
        if input_path.is_dir():
            processed_files = []
            text_files = list(input_path.glob('*.txt'))  # Add more extensions if needed
            
            # Process each file with progress bar
            for file_path in tqdm(text_files, desc="Processing local files"):
                try:
                    # Simplified output path - just append _processed to the original name
                    output_file = output_path.parent / f"local_processed_{file_path.name}"
                    
                    with open(file_path, 'r', encoding='utf-8') as infile, \
                         open(output_file, 'w', encoding='utf-8') as outfile:
                        while chunk := infile.read(chunk_size):
                            outfile.write(chunk)
                    processed_files.append(str(output_file))
                except Exception as e:
                    logging.warning(f"Error processing file {file_path}: {str(e)}")
                    continue
            
            if processed_files:
                return processed_files[0]  # Return at least one processed file
            return None

        # If input_path is a file
        else:
            with open(input_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', encoding='utf-8') as outfile:
                while chunk := infile.read(chunk_size):
                    outfile.write(chunk)
            return str(output_path)

    except Exception as e:
        logging.error(f"Error processing local dataset {input_path}: {str(e)}")
        return None

def load_datasets(dataset_config: Dict[str, Any], cache_dir: Optional[str] = None, executor: Optional[Any] = None) -> Dict[str, Any]:
    """Load datasets with enhanced validation."""
    results = {
        'datasets': {},
        'stats': {'total_samples': 0, 'failed_loads': 0}
    }
    
    for dataset in dataset_config['datasets']:
        try:
            if dataset['type'] == 'local':
                results['datasets'][dataset['name']] = dataset['config']['path']
            elif dataset['type'] == 'huggingface':
                # Extract dataset specific parameters
                dataset_name = dataset['config']['dataset_name']
                config = dataset['config'].get('config')
                
                # Different loading logic based on dataset
                if 'medical-question-answering-datasets' in dataset_name:
                    dataset_obj = load_dataset(
                        dataset_name,
                        config,  # Pass config directly for medical dataset
                        streaming=dataset['config'].get('streaming', False),
                        split=dataset['config'].get('split', 'train'),
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                
                else:
                    # For other datasets like openwebtext
                    dataset_obj = load_dataset(
                        dataset_name,
                        streaming=dataset['config'].get('streaming', False),
                        split=dataset['config'].get('split', 'train'),
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                    
                results['datasets'][dataset['name']] = dataset_obj
                
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset['name']}: {str(e)}")
            results['stats']['failed_loads'] += 1
            continue  # Continue to next dataset instead of stopping
            
    if not results['datasets']:
        raise ValueError("No datasets were successfully loaded")
            
    return results

def process_streaming_dataset(dataset: Any, output_path: Path, chunk_size: int, dataset_name: str) -> Optional[str]:
    """Process streaming dataset with chunking."""
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            batch = []
            
            pbar = tqdm(
                dataset, 
                desc="Processing streaming dataset",
                unit=" samples"
            )
            
            for item in pbar:
                text = item.get('text', '').strip()
                if text:
                    batch.append(text)
                    if len(batch) >= chunk_size:
                        f.write('\n'.join(batch) + '\n')
                        batch = []
                        
                        # Memory management
                        if psutil.virtual_memory().percent > 85:
                            gc.collect()
                            
            if batch:  # Write remaining items
                f.write('\n'.join(batch) + '\n')
                
        return str(output_path)
        
    except Exception as e:
        logging.error(f"Error processing streaming dataset {dataset_name}: {str(e)}")
        return None

class WorkerManager:
    """Manages worker pools with dynamic scaling"""
    def __init__(self, initial_workers: int = None):
        self.min_workers = 1
        self.max_workers = multiprocessing.cpu_count()
        self.current_workers = initial_workers or self.max_workers
        self.memory_manager = MemoryManager()

    def adjust_workers(self) -> int:
        """Dynamically adjust worker count based on system resources"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Reduce workers under high memory or CPU pressure
        if memory.percent > 90 or cpu_percent > 90:
            self.current_workers = self.min_workers
        elif memory.percent > 80 or cpu_percent > 80:
            self.current_workers = max(self.min_workers, self.current_workers // 2)
        elif memory.percent < 60 and cpu_percent < 60:
            self.current_workers = min(
                self.current_workers * 2,
                self.max_workers
            )
        
        logging.info(f"Adjusted workers to {self.current_workers} "
                    f"(Memory: {memory.percent}%, CPU: {cpu_percent}%)")
        return self.current_workers

    @contextmanager
    def get_executor(self):
        """Get appropriate executor based on current conditions"""
        try:
            if self.current_workers == 1:
                yield None  # Signal to use synchronous processing
            else:
                with ProcessPoolExecutor(max_workers=self.current_workers) as executor:
                    yield executor
        finally:
            self.adjust_workers()

def main():
    """Enhanced main function with better configuration and error handling."""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(
            description='Medical text tokenizer',
            conflict_handler='resolve'  # Handle argument conflicts
        )
        
        # Training arguments group
        training_args = parser.add_argument_group('Training Arguments')
        training_args.add_argument('--local_data_path', type=str, 
                          default=str(Path.cwd() / "tokens"),
                          help="Path to store processed data and tokenizer")
        training_args.add_argument('--vocab_size', type=int, 
                          default=60000,
                          help="Vocabulary size for the tokenizer")
        training_args.add_argument('--min_freq', type=int, 
                          default=2,
                          help="Minimum frequency for BPE merges")
        training_args.add_argument('--config', type=str, 
                          default="dataset_config.yaml",
                          help="Dataset configuration file path")
        training_args.add_argument('--log', type=str, 
                          default="tokenizer.log",
                          help="Log file path")
        training_args.add_argument('--chunk_size', type=int, 
                          default=10000,
                          help="Chunk size for processing")
        training_args.add_argument('--workers', type=int, 
                          default=8,
                          help="Maximum number of workers for parallel processing")

        args = parser.parse_args()
        
        # Setup logging
        setup_logging(args.log)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """Validate dataset configuration."""
    if not isinstance(config, dict):
        logging.error(f"Invalid configuration type: {type(config)}")
        return False

    required_fields = {
        'openwebtext': {'name', 'download_config'},
        'huggingface': {'name', 'huggingface_id'},
        'local': {'name', 'path', 'preprocessing'}
    }
    
    try:
        dataset_type = config.get('name')
        if not dataset_type:
            logging.error("Dataset configuration missing 'name' field")
            return False
            
        required = required_fields.get(dataset_type, {'name', 'preprocessing'})
        missing = [field for field in required if field not in config]
        
        if missing:
            logging.error(f"Dataset {dataset_type} missing required fields: {missing}")
            return False
            
        # Validate preprocessing configuration
        if 'preprocessing' not in config:
            logging.error(f"Dataset {dataset_type} missing preprocessing configuration")
            return False
            
        preproc = config['preprocessing']
        if not isinstance(preproc, dict):
            logging.error(f"Invalid preprocessing configuration type for {dataset_type}")
            return False
            
        # Validate specific fields
        if dataset_type == 'local':
            path = Path(config['path'])
            if not path.exists():
                logging.error(f"Local dataset path does not exist: {path}")
                return False
                
        return True
        
    except Exception as e:
        logging.error(f"Error validating dataset config: {str(e)}")
        return False

def load_datasets(dataset_configs: List[Dict[str, Any]], cache_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Load datasets with enhanced validation."""
    results = {
        'datasets': {},
        'stats': {'total_samples': 0, 'failed_loads': 0},
        'metadata': {}
    }
    
    # Validate configurations first
    valid_configs = [
        config for config in dataset_configs 
        if validate_dataset_config(config)
    ]
    
    if not valid_configs:
        raise ValueError("No valid dataset configurations provided")
    
    # Continue with loading using valid configurations
    with tqdm(total=len(valid_configs), desc="Loading datasets") as pbar:
        for config in valid_configs:
            try:
                dataset_name = config['name']
                dataset = load_dataset_by_type(config, cache_dir)
                
                if dataset is not None:
                    results['datasets'][dataset_name] = dataset
                    # Update statistics...
                    
            except Exception as e:
                logging.error(f"Error loading dataset {config['name']}: {str(e)}")
                results['stats']['failed_loads'] += 1
            finally:
                pbar.update(1)
                
    return results

def load_dataset_by_type(config: Dict[str, Any], cache_dir: Optional[str] = None) -> Optional[Any]:
    """Load dataset based on its type with streaming enabled for large datasets."""
    dataset_type = config['name']
    
    try:
        if dataset_type == 'openwebtext':
            return load_dataset(
                'openwebtext',
                streaming=True,  # Enable streaming
                **config['download_config'],
                trust_remote_code=True
            )
            
        elif dataset_type == 'huggingface':
            return load_dataset(
                config['huggingface_id'],
                config.get('subset'),
                streaming=True,  # Enable streaming
                cache_dir=cache_dir
            )
            
        elif dataset_type == 'local':
            return load_local_dataset(config)
            
        else:
            logging.warning(f"Unsupported dataset type: {dataset_type}")
            return None
            
    except Exception as e:
        logging.error(f"Error loading {dataset_type} dataset: {str(e)}")
        return None

def load_local_dataset(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load dataset from local directory."""
    try:
        data_path = Path(config['path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
            
        dataset = {'text': []}
        for file_path in data_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in {'.txt', '.json', '.csv'}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    dataset['text'].extend(f.readlines())
                    
        return dataset
        
    except Exception as e:
        logging.error(f"Error loading local dataset: {str(e)}")
        return None

def process_streaming_dataset(dataset, output_path: Path, chunk_size: int, dataset_name: str):
    """Process a streaming dataset in chunks."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            buffer = []
            
            # Handle both IterableDataset and dict types
            if isinstance(dataset, dict):
                items = dataset.get('text', [])
            else:
                items = dataset
                
            for item in items:
                try:
                    # Extract text from item depending on its type
                    if isinstance(item, dict):
                        text = item.get('text', '')
                    elif isinstance(item, str):
                        text = item
                    else:
                        continue
                        
                    if text and isinstance(text, str):
                        buffer.append(text)
                        
                        if len(buffer) >= chunk_size:
                            f.write('\n'.join(buffer) + '\n')
                            f.flush()
                            buffer = []
                            
                            # Memory management
                            if psutil.virtual_memory().percent > 85:
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                
                except Exception as e:
                    logging.warning(f"Error processing item in streaming dataset: {str(e)}")
                    continue
                    
            # Write remaining buffer
            if buffer:
                f.write('\n'.join(buffer) + '\n')
                
        return str(output_path)
        
    except Exception as e:
        logging.error(f"Error processing streaming dataset: {str(e)}")
        if output_path.exists():
            output_path.unlink()
        return None

def process_regular_dataset(dataset, output_path: Path, chunk_size: int, dataset_name: str):
    """Process a regular (non-streaming) dataset."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            buffer = []
            
            # Process items in chunks
            for i in range(0, len(dataset), chunk_size):
                chunk = dataset[i:i + chunk_size]
                
                for item in chunk:
                    if isinstance(item, dict):
                        text = item.get('text', '')
                    else:
                        text = str(item)
                        
                    if text and isinstance(text, str):
                        buffer.append(text)
                        
                # Write buffer
                if buffer:
                    f.write('\n'.join(buffer) + '\n')
                    f.flush()
                    buffer = []
                    
                # Memory management
                if psutil.virtual_memory().percent > 85:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            return str(output_path)
            
    except Exception as e:
        logging.error(f"Error processing regular dataset: {str(e)}")
        if output_path.exists():
            output_path.unlink()
        return None

def check_memory_usage():
    """Check memory usage and force cleanup if needed."""
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)  # Give system time to free memory
        return True
    return False

if __name__ == "__main__":
    main()
    