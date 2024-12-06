import os
import gc
import yaml
import glob
import argparse
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tokens.tokenizer import Tokenizer
from datetime import datetime
from tqdm.auto import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import wandb
import threading
from queue import Queue
from functools import wraps  # Fixing missing import
import model  # Importing model components from model.py
from pineconedb import save_embeddings_to_pinecone
from torch.utils.tensorboard import SummaryWriter
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
from torch.optim import AdamW
from collections import OrderedDict
import pickle
import shutil


## Remember to fix the tokenizer path, update it. 

# Load environment variables for Pinecone API key
load_dotenv('api.env')

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self._validate_config()

    def load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate the loaded configuration file for required sections and types."""
        # Required sections
        required_sections = ['model', 'data', 'tokenizer', 'logging', 'checkpointing', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Type validation for specific configuration values
        required_types = {
            'model.d_model': int,
            'model.src_seq_len': int,
            'training.learning_rate': float,
        }
        
        for path, expected_type in required_types.items():
            value = self._get_nested_value(path)
            if not isinstance(value, expected_type):
                raise ValueError(f"Invalid type for {path}: expected {expected_type}, got {type(value).__name__}")

    def _get_nested_value(self, path: str):
        """Get a nested value from the configuration dictionary using dot notation."""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if key in value:
                value = value[key]
            else:
                raise ValueError(f"Missing key '{key}' in configuration path '{path}'")
        return value
    
def setup_logging(config: Dict[str, Any]):
    log_dir = config['logging']['save_dir']
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    log_level = getattr(logging, config['logging'].get('level', 'INFO').upper())

    logging.basicConfig(level=log_level, filename=log_file,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

# Metrics Tracker
class MetricsTracker:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_scores = []
        self.rouge_scores = defaultdict(list)

    def update_metrics(self, reference: List[str], hypothesis: List[str]):
        bleu = sentence_bleu([reference], hypothesis)
        self.bleu_scores.append(bleu)
        rouge = self.scorer.score(" ".join(reference), " ".join(hypothesis))
        for key, value in rouge.items():
            self.rouge_scores[key].append(value.fmeasure)

    def get_average_metrics(self) -> Dict[str, float]:
        avg_bleu = sum(self.bleu_scores) / len(self.bleu_scores) if self.bleu_scores else 0.0
        avg_rouge = {key: sum(values) / len(values) if values else 0.0 for key, values in self.rouge_scores.items()}
        return {'avg_bleu': avg_bleu, **avg_rouge}

# Model Setup
class ModelManager:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

    def initialize_model(self) -> Tuple[nn.Module, Tokenizer]:
        """Initialize the Transformer model and tokenizer."""
        try:
            tokenizer_path = self.config['tokenizer']['load_path']
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logging.info(f"Tokenizer loaded from {tokenizer_path}")

            # Initialize the Transformer model using the build function from model.py
            src_vocab_size = tokenizer.get_vocab_size()
            tgt_vocab_size = src_vocab_size  # Assuming src and tgt vocab sizes are the same
            transformer = model.build_unified_transformer(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                src_seq_len=self.config['model']['src_seq_len'],
                tgt_seq_len=self.config['model']['tgt_seq_len'],
                d_model=self.config['model']['d_model']
            ).to(self.device)

            return transformer, tokenizer
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise
# Security Utilities
class SecurityUtils:
    @staticmethod
    def validate_input_data(texts: List[str], max_length: int = 1000000) -> None:
        """Validate input text data for security concerns."""
        for text in texts:
            if len(text) > max_length:
                raise ValueError(f"Input text exceeds maximum length of {max_length}")
            if not text.isprintable():
                raise ValueError("Input text contains non-printable characters")
            SecurityUtils._check_for_suspicious_patterns(text)
    
    @staticmethod
    def _check_for_suspicious_patterns(text: str) -> None:
        suspicious_patterns = [
            "<?php", "<%", "<script",
            "SELECT.*FROM", "DELETE.*FROM", "DROP.*TABLE",
            "../", "..\\", "/**/"
        ]
        for pattern in suspicious_patterns:
            if pattern.lower() in text.lower():
                raise ValueError(f"Suspicious pattern detected: {pattern}")

# Data Validator
class DataValidator:
    @staticmethod
    def validate_sequence_lengths(sequences: List[List[int]], max_length: int):
        invalid_sequences = [i for i, seq in enumerate(sequences) if len(seq) > max_length]
        if invalid_sequences:
            raise ValueError(f"Found {len(invalid_sequences)} sequences exceeding max length")

    @staticmethod
    def check_data_distribution(sequences: List[List[int]]) -> Dict[str, float]:
        lengths = [len(seq) for seq in sequences]
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths)
        }

# Validation Loop
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", dynamic_ncols=True):
            outputs = model(batch['input_ids'].to(device), batch['target_ids'].to(device))
            total_val_loss += outputs.loss.item()
    return total_val_loss / len(val_loader)

# Memory Monitor
class MemoryMonitor:
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0
        self.memory_logs = []

    def log_memory(self, step: int):
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(self.device)
            self.peak_memory = max(self.peak_memory, current_memory)
            self.memory_logs.append((step, current_memory))
            
            # Alert if memory usage is too high (90% of available memory)
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            if current_memory > 0.9 * total_memory:
                logging.warning(f"High memory usage detected: {current_memory / 1e9:.2f}GB")

# Data Manager for handling data loading
class DataManager:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.tokenizer: Optional[Tokenizer] = None

    def load_openwebtext(self) -> List[str]:
        from datasets import load_dataset
        dataset = load_dataset("openwebtext", split=f"train[:{self.config['data']['max_samples']}]")
        return [item['text'] for item in dataset]

    def load_medical_datasets(self) -> List[str]:
        from datasets import load_dataset
        medical_data = []
        try:
            pubmed_qa_data = load_dataset("pubmed_qa", "pqa_artificial", split="train")
            medical_data.extend([item["question"] + " " + item["context"] for item in pubmed_qa_data])
        except Exception as e:
            logging.warning(f"Error loading PubMed QA dataset: {e}")
        return medical_data[:self.config['data']['max_samples']]

    def load_local_data(self, directory: str) -> List[str]:
        texts = []
        for file_name in glob.glob(os.path.join(directory, "*.txt")):
            try:
                with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
                    texts.extend(f.readlines())
            except Exception as e:
                logging.warning(f"Error reading file {file_name}: {e}")
        return texts[:self.config['data']['max_samples']]

    def load_data_parallel(self, texts: List[str], num_workers: int = 4) -> List[List[int]]:
        chunk_size = len(texts) // num_workers
        with Pool(num_workers) as pool:
            chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            results = pool.map(self._process_chunk, chunks)
        return [item for sublist in results for item in sublist]

    def _process_chunk(self, texts: List[str]) -> List[List[int]]:
        processed = []
        for text in texts:
            try:
                tokens = self.tokenizer.encode(text).ids
                if len(tokens) >= self.config['model']['src_seq_len']:
                    processed.append(tokens[:self.config['model']['src_seq_len']])
            except Exception as e:
                logging.warning(f"Error processing text: {str(e)}")
                continue
        return processed

# Async Prefetch Data Loader
class AsyncPrefetchDataLoader:
    def __init__(self, dataloader: DataLoader, device: torch.device, num_prefetch: int = 3):
        self.dataloader = dataloader
        self.device = device
        self.num_prefetch = num_prefetch
        self.queue = Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_data, daemon=True)
        self.prefetch_thread.start()

    def _prefetch_data(self):
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break
                processed_batch = {k: v.to(self.device, non_blocking=True) 
                                 if isinstance(v, torch.Tensor) else v 
                                 for k, v in batch.items()}
                self.queue.put(processed_batch)
        except Exception as e:
            logging.error(f"Error in prefetch thread: {e}")
        finally:
            self.queue.put(None)  # Signal end of data

    def __iter__(self):
        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch

    def __del__(self):
        self.stop_event.set()
        if hasattr(self, 'prefetch_thread'):
            self.prefetch_thread.join()

class MemoryManager:
    @staticmethod
    def clean_memory(aggressive: bool = False):
        """
        Clean up memory across CPU and GPU.

        Args:
            aggressive (bool): If True, attempts a more aggressive GPU memory clean-up.
        """
        # Clean up Python garbage collection
        gc.collect()
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.ipc_collect()
        logging.info("Memory cleaned up successfully.")

    @staticmethod
    def log_memory_usage():
        """
        Log the current and peak memory usage on the GPU.
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Peak allocated
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Cached memory
            logging.info(
                f"GPU Memory Usage: Allocated: {allocated:.2f} GB, "
                f"Max Allocated: {max_allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )
        else:
            logging.info("No GPU available to log memory usage.")

    @staticmethod
    def get_memory_threshold(threshold_ratio: float = 0.85) -> Optional[float]:
        """
        Get memory threshold for triggering alerts.

        Args:
            threshold_ratio (float): Percentage threshold of total GPU memory to trigger alerts.

        Returns:
            float: Threshold value for GPU memory usage.
        """
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return threshold_ratio * total_memory
        return None

    @staticmethod
    def monitor_memory_usage(threshold_ratio: float = 0.85) -> None:
        """
        Check and log whether current GPU memory usage exceeds a given threshold.

        Args:
            threshold_ratio (float): Percentage threshold of total GPU memory to trigger alerts.
        """
        if torch.cuda.is_available():
            threshold = MemoryManager.get_memory_threshold(threshold_ratio)
            if torch.cuda.memory_allocated() > threshold:
                logging.warning(f"Memory usage exceeds {threshold_ratio * 100:.0f}% of GPU capacity!")
                MemoryManager.clean_memory(aggressive=True)

    @staticmethod
    def monitor_memory(func):
        """
        Decorator to log memory usage before and after function execution.

        Args:
            func (callable): The function to be wrapped.

        Returns:
            callable: Wrapped function that logs memory usage.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            MemoryManager.log_memory_usage()  # Log before execution
            
            try:
                result = func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error("Out of memory error. Cleaning memory...")
                    MemoryManager.clean_memory(aggressive=True)
                    raise e
                else:
                    raise e
            
            MemoryManager.log_memory_usage()  # Log after execution
            return result
        return wrapper

    @staticmethod
    def handle_oom(func):
        """
        Decorator to handle out-of-memory errors by cleaning memory and retrying the operation.

        Args:
            func (callable): The function to be wrapped.

        Returns:
            callable: Wrapped function that retries execution after OOM error.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error("Out of memory error detected. Cleaning memory and attempting recovery...")
                    MemoryManager.clean_memory(aggressive=True)
                    # Optionally, clear optimizer gradients if passed
                    optimizer = kwargs.get('optimizer')
                    if optimizer:
                        optimizer.zero_grad()
                    return None  # Allow retry logic in training loop
                raise e
        return wrapper

    @staticmethod
    def log_cpu_memory_usage():
        """
        Log the current memory usage for the CPU.
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        rss = memory_info.rss / (1024 ** 2)  # Resident Set Size in MB
        vms = memory_info.vms / (1024 ** 2)  # Virtual Memory Size in MB

        logging.info(f"CPU Memory Usage: RSS: {rss:.2f} MB, VMS: {vms:.2f} MB")

    @staticmethod
    def monitor_memory_decorator(func):
        """
        Decorator to log both CPU and GPU memory usage before and after a function's execution.

        Args:
            func (callable): The function to be wrapped.

        Returns:
            callable: Wrapped function that logs CPU and GPU memory usage.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info("Monitoring memory before execution...")
            MemoryManager.log_cpu_memory_usage()
            MemoryManager.log_memory_usage()  # Log GPU memory
            
            try:
                result = func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error("Out of memory error detected during function execution. Attempting recovery...")
                    MemoryManager.clean_memory(aggressive=True)
                    raise e
                else:
                    raise e
            
            logging.info("Monitoring memory after execution...")
            MemoryManager.log_cpu_memory_usage()
            MemoryManager.log_memory_usage()  # Log GPU memory
            return result
        return wrapper

class CheckpointManager:
    def __init__(self, save_dir: str, max_checkpoints: int = 3):
        """
        Manages saving and cleaning up model checkpoints.

        Args:
            save_dir (str): Directory to save checkpoints.
            max_checkpoints (int): Maximum number of checkpoints to retain.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        # Initialize checkpoints from existing files
        self.checkpoints = sorted(self.save_dir.glob("*.pt"), key=os.path.getmtime)

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, metrics: Dict[str, float], best: bool = False) -> None:
        """
        Saves a checkpoint and manages older ones.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): Optimizer to save.
            epoch (int): The epoch number.
            loss (float): Loss value.
            metrics (Dict[str, float]): Additional metrics to save.
            best (bool): Whether this is the best checkpoint.
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt" if not best else f"best_checkpoint.pt"
        checkpoint_path = self.save_dir / checkpoint_name

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Check available disk space before saving
            if shutil.disk_usage(self.save_dir).free < 500 * 1024 * 1024:  # Less than 500MB
                logging.warning("Low disk space. Checkpoint might fail to save.")

            torch.save(checkpoint_data, checkpoint_path)
            if not best:
                self.checkpoints.append(checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")

            # Remove old checkpoints if exceeding the limit
            self._cleanup_old_checkpoints()

        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            raise

    def _cleanup_old_checkpoints(self) -> None:
        """
        Removes the oldest checkpoints if the total exceeds `max_checkpoints`.
        """
        while len(self.checkpoints) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoints.pop(0)
            try:
                oldest_checkpoint.unlink()
                logging.info(f"Removed old checkpoint: {oldest_checkpoint}")
            except Exception as e:
                logging.warning(f"Failed to remove checkpoint {oldest_checkpoint}: {e}")

    def load_latest_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location: Optional[Any] = None) -> Tuple[Optional[int], Optional[Dict[str, float]]]:
        """
        Loads the latest checkpoint.

        Args:
            model (torch.nn.Module): The model to load the state dict into.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load the state dict into.
            map_location (optional): Map location for loading checkpoints (useful for multi-GPU or CPU).

        Returns:
            Tuple[int, Dict[str, float]]: The epoch and metrics from the checkpoint.
        """
        if not self.checkpoints:
            logging.warning("No checkpoints available to load.")
            return None, None

        latest_checkpoint = self.checkpoints[-1]
        logging.info(f"Loading checkpoint: {latest_checkpoint}")

        try:
            checkpoint_data = torch.load(latest_checkpoint, map_location=map_location)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            return checkpoint_data.get('epoch', 0), checkpoint_data.get('metrics', {})
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise

    def load_checkpoint_by_path(self, checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location: Optional[Any] = None) -> Tuple[Optional[int], Optional[Dict[str, float]]]:
        """
        Load a checkpoint from a specified file path.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model (torch.nn.Module): The model to load the state dict into.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load the state dict into.
            map_location (optional): Map location for loading checkpoints (useful for multi-GPU or CPU).

        Returns:
            Tuple[int, Dict[str, float]]: The epoch and metrics from the checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint path does not exist: {checkpoint_path}")
            return None, None

        logging.info(f"Loading checkpoint from specified path: {checkpoint_path}")

        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            return checkpoint_data.get('epoch', 0), checkpoint_data.get('metrics', {})
        except Exception as e:
            logging.error(f"Failed to load checkpoint from path: {e}")
            raise

    def load_best_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location: Optional[Any] = None) -> Tuple[Optional[int], Optional[Dict[str, float]]]:
        """
        Loads the best checkpoint (if available).

        Args:
            model (torch.nn.Module): The model to load the state dict into.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load the state dict into.
            map_location (optional): Map location for loading checkpoints (useful for multi-GPU or CPU).

        Returns:
            Tuple[int, Dict[str, float]]: The epoch and metrics from the checkpoint.
        """
        best_checkpoint_path = self.save_dir / "best_checkpoint.pt"
        if not best_checkpoint_path.exists():
            logging.warning("Best checkpoint does not exist.")
            return None, None

        logging.info(f"Loading best checkpoint from: {best_checkpoint_path}")

        try:
            checkpoint_data = torch.load(best_checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            return checkpoint_data.get('epoch', 0), checkpoint_data.get('metrics', {})
        except Exception as e:
            logging.error(f"Failed to load best checkpoint: {e}")
            raise

# Model Manager for handling model initialization
class ModelManager:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(config['checkpointing']['save_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.scaler = GradScaler()

    def initialize_model(self) -> Tuple[nn.Module, Tokenizer]:
        """Initialize model and tokenizer with enhanced error handling"""
        try:
            tokenizer = Tokenizer.from_file(self.config['tokenizer']['load_path'])
            logging.info(f"Tokenizer loaded from {self.config['tokenizer']['load_path']}")
            
            src_vocab_size = tokenizer.get_vocab_size()
            model = torch.jit.script(model.build_unified_transformer(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=src_vocab_size,
                src_seq_len=self.config['model']['src_seq_len'],
                tgt_seq_len=self.config['model']['src_seq_len'],
                d_model=self.config['model']['d_model']
            )).to(self.device)

            return model, tokenizer
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise

def handle_oom(func):
    """Decorator to handle out-of-memory errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("Out of memory error detected. Clearing memory and attempting recovery...")
                MemoryManager.clean_memory(aggressive=True)
                optimizer = kwargs.get('optimizer')
                if optimizer:
                    optimizer.zero_grad()
                return None  # Allow retry logic
            raise e
    return wrapper

@handle_oom
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_manager: CheckpointManager,
    writer: SummaryWriter
):
    model.train()
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    best_val_loss = float('inf')
    patience_counter = 0
    val_frequency = config['training'].get('val_frequency', 1)  # Default: validate every epoch

    for epoch in range(config['training']['epochs']):
        logging.info(f"Starting epoch {epoch + 1}/{config['training']['epochs']}")
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(progress_bar):
            # Forward and backward pass with mixed precision
            with autocast(enabled=config['training']['mixed_precision']):
                outputs = model(batch['input_ids'].to(device), batch['target_ids'].to(device))
                loss = outputs.loss / config['training']['gradient_accumulation_steps']

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

            # Log batch loss
            if (batch_idx + 1) % config['logging']['log_interval'] == 0:
                writer.add_scalar('Loss/Batch', loss.item(), epoch * len(train_loader) + batch_idx)

        # Average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        logging.info(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

        # Validation
        if (epoch + 1) % val_frequency == 0:
            val_loss = validate(model, val_loader, device)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            logging.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

            # Checkpoint and early stopping
            if val_loss < best_val_loss - config['training']['early_stopping_threshold']:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_manager.save_checkpoint(model, optimizer, epoch, metrics={'val_loss': val_loss})
                logging.info(f"Checkpoint saved for epoch {epoch + 1}")
            else:
                patience_counter += 1
                if patience_counter >= config['training']['patience']:
                    logging.info("Early stopping triggered.")
                    break

    writer.close()
@dataclass
class TrainingConfig:
    epochs: int
    weight_decay: float
    max_grad_norm: float
    patience: int
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    warmup_steps: int = 1000
    early_stopping_threshold: float = 0.01

# Training loop with monitoring and gradient accumulation
def train_with_monitoring(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    config: TrainingConfig,
    device: torch.device,
    memory_monitor: MemoryMonitor,
    checkpoint_manager: CheckpointManager
) -> Dict[str, float]:
    
    model.train()
    scaler = GradScaler(enabled=config.mixed_precision)
    total_loss = 0.0
    
    # Training loop with error recovery
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training with Monitoring")):
        try:
            with autocast(enabled=config.mixed_precision):
                outputs = model(batch['input_ids'].to(device), batch['target_ids'].to(device))
                loss = outputs.loss / config.gradient_accumulation_steps
            
            # Gradient accumulation
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )
                
                # Log metrics
                metrics = {
                    'loss': loss.item(),
                    'grad_norm': grad_norm,
                    'lr': scheduler.get_last_lr()[0]
                }
                wandb.log(metrics)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Monitor memory usage
            if batch_idx % 100 == 0:
                memory_monitor.log_memory(batch_idx)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                MemoryManager.clean_memory(aggressive=True)
                logging.error(f"OOM error in batch {batch_idx}. Attempting recovery...")
                optimizer.zero_grad()
                continue
            raise e
    
    return {'avg_loss': total_loss / len(train_loader)}

class EmbeddingGenerator:
    def __init__(self, model: nn.Module, device: torch.device, index_name: str, cache_size: int = 100):
        self.model = model.to(device)
        self.device = device
        self.index_name = index_name
        self.cache = OrderedDict()  # Cache with LRU eviction
        self.cache_size = cache_size

    def _save_cache_to_disk(self, file_path: str = "embedding_cache.pkl"):
        """Save the cache to disk."""
        with open(file_path, "wb") as f:
            pickle.dump(self.cache, f)
        logging.info("Embedding cache saved to disk.")

    def _load_cache_from_disk(self, file_path: str = "embedding_cache.pkl"):
        """Load the cache from disk."""
        try:
            with open(file_path, "rb") as f:
                self.cache = pickle.load(f)
            logging.info("Embedding cache loaded from disk.")
        except FileNotFoundError:
            logging.warning("No existing cache file found. Starting with an empty cache.")

    def _update_cache(self, key: str, embedding: torch.Tensor):
        """Update the cache with LRU eviction."""
        if key in self.cache:
            # Move key to the end to show it was recently used
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.cache_size:
            # Evict the least recently used item (first item in the OrderedDict)
            evicted_key, _ = self.cache.popitem(last=False)
            logging.info(f"Evicted key from cache: {evicted_key}")
        self.cache[key] = embedding

    @torch.no_grad()
    def _generate_embedding(self, input_ids: List[int]) -> torch.Tensor:
        """Generate embedding for a single input sequence."""
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with autocast(enabled=True):
            # Ensure that the forward pass returns embeddings
            output = self.model(input_tensor, input_tensor, return_embeddings=True)
        
        if isinstance(output, tuple):
            # If the model returns multiple items (e.g., logits, embeddings), handle appropriately
            embedding = output[1]  # Assuming embeddings are at index 1
        else:
            embedding = output
        
        return embedding.squeeze(0).cpu()

    def get_embedding(self, input_ids: List[int], cache_only: bool = False) -> torch.Tensor:
        """Get embedding, either from cache or dynamically generate."""
        key = str(input_ids)  # Use input sequence as a unique key
        if key in self.cache:
            logging.info("Embedding retrieved from cache.")
            return self.cache[key]

        if cache_only:
            logging.error("Embedding not found in cache and `cache_only` is True.")
            raise KeyError("Embedding not found in cache.")

        # Dynamically generate embedding and update cache
        embedding = self._generate_embedding(input_ids)
        self._update_cache(key, embedding)
        logging.info("Embedding generated dynamically and cached.")
        return embedding

    def save_embeddings_to_pinecone(self, input_ids: List[List[int]], batch_size: int = 32):
        """Generate and save embeddings to Pinecone."""
        embeddings = []
        ids = []

        for i in range(0, len(input_ids), batch_size):
            batch = input_ids[i:i + batch_size]
            batch_embeddings = []

            # Generate embeddings for each sequence in the batch
            for ids in batch:
                try:
                    embedding = self.get_embedding(ids)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logging.error(f"Error generating embedding for input_ids {ids}: {e}")
                    continue

            if batch_embeddings:
                embeddings.extend(batch_embeddings)
                ids.extend([f"embedding_{j}" for j in range(i, i + len(batch_embeddings))])

        try:
            # Save to Pinecone
            if embeddings:
                save_embeddings_to_pinecone(torch.stack(embeddings), ids, self.index_name)
                logging.info(f"Embeddings successfully saved to Pinecone for batch of size {len(ids)}")
            else:
                logging.warning("No embeddings to save to Pinecone.")
        except Exception as e:
            logging.error(f"Error saving embeddings to Pinecone: {e}")
            raise

    def save_embeddings_for_model(self, version: str = "v1", save_path: str = None):
        """
        Save generated embeddings to a file to be loaded into the model's embedding layer.

        Args:
            version (str): The version of the embeddings.
            save_path (str): The path where the embeddings should be saved. If not provided, 
                             it will be saved with a default versioned name.
        """
        if not save_path:
            save_path = f"luminalm_embeddings_{version}.pt"
        
        logging.info(f"Saving pre-trained embeddings to {save_path}")

        # Assuming that embeddings are to be saved as the full embedding matrix
        # This should match the vocab size and embedding dimension
        embeddings_matrix = torch.zeros(self.model.config.vocab_size, self.model.config.n_embd)

        for i in range(self.model.config.vocab_size):
            input_ids = [i]
            embedding = self.get_embedding(input_ids)
            embeddings_matrix[i] = embedding

        torch.save(embeddings_matrix, save_path)
        logging.info(f"Pre-trained embeddings successfully saved to {save_path}")

def main() -> None:
    try:
        # Parse arguments
        args = parse_arguments()

        # Initialize configuration and logging
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        setup_logging(config)

        # Setup WandB properly
        with open(args.config, "r") as config_file:
            wandb_config = yaml.safe_load(config_file)

        # Initialize wandb for experiment tracking
        wandb.init(project="transformer-training", config=wandb_config)
        logging.info("Initialized wandb for experiment tracking")

        # Setup device
        device = setup_device()

        # Initialize model and tokenizer
        model_manager = ModelManager(config, device)
        model, tokenizer = model_manager.initialize_model()

        # Load pre-trained embeddings if available
        if args.pretrained_embeddings:
            load_pretrained_embeddings(model, args.pretrained_embeddings)

        # Setup DataManager and load datasets
        data_manager = DataManager(config, device)
        data_manager.tokenizer = tokenizer
        sequences = load_and_process_data(data_manager, args)

        # Setup DataLoader, optimizer, and scheduler
        train_loader = create_data_loader(sequences, config, device)
        optimizer, scheduler = setup_optimizer_and_scheduler(model, config)

        # Initialize monitoring and checkpoint tools
        memory_monitor = MemoryMonitor(device)
        checkpoint_manager = CheckpointManager(config['checkpointing']['save_dir'], max_checkpoints=3)

        # Start training loop
        train_model(model, train_loader, optimizer, scheduler, config, device, memory_monitor, checkpoint_manager)

        # Generate and save embeddings after training
        generate_embeddings_and_save(model, sequences, config, device)

    except Exception as e:
        logging.error(f"Critical error during execution: {str(e)}")
        raise
    finally:
        # Cleanup
        wandb.finish()
        MemoryManager.clean_memory(aggressive=True)

# Argument Parsing
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train hybrid transformer-based embeddings model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training')
    parser.add_argument('--local_data_dir', type=str, help='Path to local data directory')
    return parser.parse_args()

# Device Setup
def setup_device() -> torch.device:
    """Setup the device (CPU or GPU) to be used for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

# Load and Process Data
def load_and_process_data(data_manager: DataManager, args: argparse.Namespace) -> List[List[int]]:
    """Load and process data from various sources."""
    logging.info("Loading datasets...")
    try:
        openwebtext_data = data_manager.load_openwebtext()
        medical_data = data_manager.load_medical_datasets()
        local_data = data_manager.load_local_data(args.local_data_dir) if args.local_data_dir else []

        all_texts = openwebtext_data + medical_data + local_data
        SecurityUtils.validate_input_data(all_texts)
        sequences = data_manager.load_data_parallel(all_texts)

        # Validate sequence lengths
        DataValidator.validate_sequence_lengths(sequences, max_length=data_manager.config['model']['src_seq_len'])
        
        return sequences
    except Exception as e:
        logging.error(f"Error during data loading and processing: {str(e)}")
        raise

# Create DataLoader
def create_data_loader(sequences: List[List[int]], config: Dict[str, Any], device: torch.device) -> DataLoader:
    """Create the DataLoader for training."""
    dataset = AsyncPrefetchDataLoader(DataLoader(sequences, batch_size=config['model']['batch_size']), device)
    logging.info("DataLoader created successfully")
    return dataset

# Setup Optimizer and Scheduler
def setup_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, CosineAnnealingLR]:
    """Setup optimizer and learning rate scheduler."""
    optimizer = AdamW(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['model']['epochs'])
    logging.info("Optimizer and scheduler initialized")
    return optimizer, scheduler

# Training Loop
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    config: Dict[str, Any],
    device: torch.device,
    memory_monitor: MemoryMonitor,
    checkpoint_manager: CheckpointManager
) -> None:
    """Training loop with enhanced monitoring and checkpointing."""
    best_val_loss = float('inf')
    patience_counter = 0
    logging.info("Starting training loop")

    for epoch in range(config['model']['epochs']):
        logging.info(f"Starting epoch {epoch + 1}/{config['model']['epochs']}")
        avg_loss = train_with_monitoring(
            model, train_loader, optimizer, scheduler, TrainingConfig(**config['training']), device, memory_monitor, checkpoint_manager
        )

        logging.info(f"Epoch {epoch + 1} average loss: {avg_loss['avg_loss']}")
        if avg_loss['avg_loss'] < best_val_loss:
            best_val_loss = avg_loss['avg_loss']
            patience_counter = 0
            checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, avg_loss['avg_loss'],
                metrics={'avg_loss': avg_loss['avg_loss']}
            )
            logging.info(f"Checkpoint saved for epoch {epoch + 1}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['training']['patience']:
            logging.info("Early stopping triggered.")
            break

# Generate and Save Embeddings
def generate_embeddings_and_save(model: nn.Module, sequences: List[List[int]], config: Dict[str, Any], device: torch.device) -> None:
    """Generate and save embeddings using the trained model."""
    logging.info("Generating and saving embeddings to Pinecone...")
    embedding_generator = EmbeddingGenerator(model, device, config['pinecone']['index_name'])
    embedding_generator.save_embeddings_to_pinecone(sequences)
    logging.info("Embeddings saved successfully")

# Load Pre-trained Embeddings
def load_pretrained_embeddings(model: nn.Module, embeddings_path: str):
    """Load pre-trained embeddings into the model's embedding layer."""
    try:
        logging.info(f"Loading pre-trained embeddings from {embeddings_path}")
        pretrained_embeddings = torch.load(embeddings_path)
        
        # Assuming the embeddings are stored in a compatible format
        model.transformer['wte'].weight.data.copy_(pretrained_embeddings)
        logging.info("Pre-trained embeddings loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load pre-trained embeddings: {e}")
        raise

# Entry Point
if __name__ == "__main__":
    main()

