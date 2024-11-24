import os
import gc
import json
import yaml
import argparse
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_xla.core import xla_model
from torch_xla.utils.utils import tpu_available
from tokenizers import Tokenizer
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from multiprocessing import Pool
import pinecone  # Ensure pinecone-client is installed

import model  # Assuming model.py is in the same directory or accessible in PYTHONPATH

def initialize_device(config: Dict[str, Any]):
    """Initialize the appropriate device based on the configuration."""
    if config['device']['type'] == "tpu" or (config['device']['type'] == "auto" and tpu_available()):
        logging.info("Using TPU as the training device.")
        device = xla_model.xla_device()
    elif config['device']['type'] == "gpu" or (config['device']['type'] == "auto" and torch.cuda.is_available()):
        logging.info("Using GPU as the training device.")
        device = torch.device("cuda")
    else:
        logging.info("Using CPU as the training device.")
        device = torch.device("cpu")
    return device

def setup_logging(config: Dict[str, Any]):
    log_level = getattr(logging, config['logging']['level'].upper(), logging.INFO)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['logging']['save_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            return self.get_default_config()

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            'model': {
                'd_model': 512,
                'src_seq_len': 512,
                'batch_size': 128,
                'learning_rate': 5e-5,
                'epochs': 3,
                'patience': 3
            },
            'data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'max_samples': 100000
            },
            'training': {
                'use_mixed_precision': True,
                'gradient_accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'warmup_steps': 1000,
                'validation_steps': None,
                'gradient_noise_std': 0.0
            },
            'tokenizer': {
                'save_path': "embeddings/tokenizer.json",
                'load_path': "embeddings/tokenizer.json"
            },
            'logging': {
                'level': 'INFO',
                'save_dir': 'logs',
                'metrics_file': 'metrics.json'
            },
            'checkpointing': {
                'save_dir': 'checkpoints',
                'save_frequency': 1,
                'keep_best_n': 3
            },
            'visualization': {
                'plot_dir': 'plots',
                'sample_size': 100000,
                'embedding_dims': 3
            },
            'distributed': {
                'backend': 'nccl',
                'world_size': -1,
                'init_method': 'env://'
            },
            'pinecone': {
                'api_key': "e38016a0-15c6-4f83-aa30-9f3821a819fc",
                'environment': "us-east-1",
                'index_name': "luminalm-embeddings"
            },
            'device': {
                'type': 'auto'
            }
        }

class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def load_openwebtext(self) -> List[str]:
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                "openwebtext",
                split=f"train[:{self.config['data']['max_samples']}]",
                trust_remote_code=False
            )
            return [item['text'] for item in dataset]
        except Exception as e:
            logging.error(f"Error loading OpenWebText: {str(e)}")
            return []

    def load_medical_datasets(self) -> List[str]:
        from datasets import load_dataset
        datasets_to_load = [
            {"name": "pubmed_qa", "config": "pqa_artificial", "split": "train"},
            {"name": "scicite", "split": "train"},
            # Add others as they become accessible
        ]
        texts = []

        for dataset_info in datasets_to_load:
            try:
                name = dataset_info["name"]
                config_name = dataset_info.get("config")
                split = dataset_info.get("split", "train")
                
                if config_name:
                    dataset = load_dataset(name, config_name, split=split)
                else:
                    dataset = load_dataset(name, split=split)
                
                # Extract the relevant column(s)
                if "text" in dataset.column_names:
                    texts.extend(dataset["text"])
                elif "sentence" in dataset.column_names:
                    texts.extend(dataset["sentence"])
                elif all(col in dataset.column_names for col in ["question", "context"]):
                    texts.extend([f"{q.strip()} {c.strip()}" for q, c in zip(dataset["question"], dataset["context"])])
                else:
                    logging.warning(f"No relevant text columns found in {name}. Skipping dataset.")
                
                logging.info(f"Loaded {len(dataset)} examples from {name}")

            except Exception as e:
                logging.error(f"Error processing dataset {name}: {str(e)}")

        return texts[:self.config['data']['max_samples']]

    def load_local_data(self, directory: str) -> List[str]:
        texts = []
        try:
            file_list = [os.path.join(directory, file) 
                         for file in os.listdir(directory) 
                         if file.endswith(".txt")]
            for file_name in file_list:
                if not os.path.exists(file_name):
                    logging.warning(f"File not found: {file_name}")
                    continue
                try:
                    with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
                        texts.extend(f.readlines())
                except Exception as e:
                    logging.error(f"Error reading file {file_name}: {str(e)}")
        except Exception as e:
            logging.error(f"Error accessing directory {directory}: {str(e)}")
        return texts[:self.config['data']['max_samples']]

    @staticmethod
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]

        input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
        target_ids_padded = rnn_utils.pad_sequence(target_ids, batch_first=True, padding_value=0)

        return {"input_ids": input_ids_padded, "target_ids": target_ids_padded}

class CustomDataset(Dataset):
    def __init__(self, sequences: List[List[int]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }

class ModelManager:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.checkpoint_dir = config['checkpointing']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def initialize_model(self) -> Tuple[nn.Module, Tokenizer]:
        try:
            tokenizer_path = self.config['tokenizer']['load_path']
            tokenizer = Tokenizer.from_file(tokenizer_path)
            logging.info(f"Tokenizer loaded from {tokenizer_path}")
            src_vocab_size = tokenizer.get_vocab_size()
            tgt_vocab_size = src_vocab_size

            transformer_model = model.build_transformer(
                src_vocab_size,
                tgt_vocab_size,
                src_seq_len=self.config['model']['src_seq_len'],
                tgt_seq_len=self.config['model']['src_seq_len'],
                d_model=self.config['model']['d_model']
            ).to(self.device)
            
            return transformer_model, tokenizer
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, loss: float, is_best: bool = False) -> None:
        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(model.state_dict(), best_path)
                logging.info(f"Best model saved: {best_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        checkpoint_path: str) -> Tuple[nn.Module, torch.optim.Optimizer, int, float]:
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            return model, optimizer, epoch, loss
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            raise

    def save_tokenizer(self, tokenizer: Tokenizer) -> None:
        save_path = self.config['tokenizer']['save_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokenizer.save(save_path)
        logging.info(f"Tokenizer saved to {save_path}")

class Trainer:
    def __init__(self, config: Dict[str, Any], model: nn.Module, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.scaler = torch.amp.GradScaler(enabled=config['training']['use_mixed_precision'])
        self.validation_steps = config['training'].get('validation_steps')
        self.gradient_noise_std = config['training'].get('gradient_noise_std', 0.0)

    def add_gradient_noise(self):
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.gradient_noise_std
                param.grad.add_(noise)

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, scheduler=None, epoch_num: int = 0) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        total_perplexity = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            with torch.amp.autocast(enabled=self.config['training']['use_mixed_precision']):
                outputs = self.model(input_ids, target_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                perplexity = torch.exp(loss)

            # Gradient scaling and accumulation
            self.scaler.scale(loss).backward()

            if self.gradient_noise_std > 0:
                self.add_gradient_noise()

            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()
            total_perplexity += perplexity.item()
            _, predicted = torch.max(outputs, -1)
            correct_predictions += (predicted == target_ids).sum().item()
            total_predictions += target_ids.numel()

            # Memory management
            del outputs, loss
            torch.cuda.empty_cache()
            if batch_idx % 100 == 0:
                gc.collect()

            # Validation at regular steps
            if self.validation_steps and batch_idx % self.validation_steps == 0 and batch_idx > 0:
                val_loss, val_accuracy = self.validate(val_loader, criterion)
                logging.info(f"Intermediate Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        avg_perplexity = total_perplexity / len(train_loader)
        
        return avg_loss, accuracy, avg_perplexity

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_predictions = 0

        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            outputs = self.model(input_ids, target_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, -1)
            correct_val_predictions += (predicted == target_ids).sum().item()
            total_val_predictions += target_ids.numel()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val_predictions / total_val_predictions
        return avg_val_loss, val_accuracy

class EmbeddingGenerator:
    def __init__(self, model: nn.Module, device: torch.device, index):
        self.model = model
        self.device = device
        self.index = index

    @torch.no_grad()
    def generate_embeddings(self, input_ids_batches: List[List[int]], 
                            batch_size: int = 32, chunk_size: int = 1000) -> torch.Tensor:
        self.model.eval()
        all_embeddings = []
        all_ids = []
        
        for i in tqdm(range(0, len(input_ids_batches), batch_size), desc="Generating Embeddings"):
            batch = input_ids_batches[i:i + batch_size]
            input_ids = rnn_utils.pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in batch], batch_first=True, padding_value=0).to(self.device)
            
            embeddings = self.model.encode(input_ids, src_mask=None).cpu()
            all_embeddings.append(embeddings)
            
            batch_ids = [f"embedding_{i + j}" for j in range(len(embeddings))]
            all_ids.extend(batch_ids)
            
            del embeddings, input_ids
            torch.cuda.empty_cache()
            
            if i % 100 == 0:
                gc.collect()
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Save embeddings in chunks
        for start_idx in range(0, len(all_embeddings), chunk_size):
            end_idx = start_idx + chunk_size
            chunk_embeddings = all_embeddings[start_idx:end_idx]
            chunk_ids = all_ids[start_idx:end_idx]
            self.save_embeddings_to_pinecone(chunk_embeddings, chunk_ids)
        
        logging.info("All embeddings saved to PineconeDB successfully.")
        return all_embeddings

    def save_embeddings_to_pinecone(self, embeddings: torch.Tensor, ids: List[str]):
        embeddings_list = embeddings.numpy().tolist()
        vectors = list(zip(ids, embeddings_list))
        
        try:
            self.index.upsert(vectors)
            logging.info(f"Saved {len(vectors)} embeddings to PineconeDB.")
        except Exception as e:
            logging.error(f"Error saving embeddings to PineconeDB: {str(e)}")

class Visualizer:
    @staticmethod
    def plot_metrics(metrics: Dict[str, List[float]], config: Dict[str, Any]):
        save_dir = config['visualization']['plot_dir']
        os.makedirs(save_dir, exist_ok=True)
        for metric_name, values in metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(values, label=metric_name)
            plt.title(f'Training {metric_name}')
            plt.xlabel('Epochs')
            plt.ylabel(metric_name)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{metric_name}.png'))
            plt.close()

    @staticmethod
    def plot_embeddings_3d(embeddings: torch.Tensor, method: str, config: Dict[str, Any]):
        save_dir = config['visualization']['plot_dir']
        os.makedirs(save_dir, exist_ok=True)
        sample_size = min(len(embeddings), config['visualization']['sample_size'])
        embeddings_sample = embeddings[:sample_size]

        try:
            if method == "PCA":
                reducer = PCA(n_components=config['visualization']['embedding_dims'])
            elif method == "t-SNE":
                reducer = TSNE(n_components=config['visualization']['embedding_dims'], random_state=42)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
                
            reduced_embeddings = reducer.fit_transform(embeddings_sample)

            if config['visualization']['embedding_dims'] == 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    reduced_embeddings[:, 2],
                    alpha=0.5,
                    c=range(len(reduced_embeddings)),
                    cmap='viridis'
                )
                plt.colorbar(scatter)
                ax.set_title(f'3D {method} Projection of Embeddings')
            else:
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1],
                    alpha=0.5,
                    c=range(len(reduced_embeddings)),
                    cmap='viridis'
                )
                plt.colorbar()
                plt.title(f'2D {method} Projection of Embeddings')

            plt.savefig(os.path.join(save_dir, f"{method}_embeddings.png"))
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting embeddings: {str(e)}")

def ensure_directories(config: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(config['tokenizer']['save_path']), exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    os.makedirs(config['checkpointing']['save_dir'], exist_ok=True)
    os.makedirs(config['visualization']['plot_dir'], exist_ok=True)

def tokenize_text(text: str, tokenizer: Tokenizer, seq_length: int) -> List[List[int]]:
    tokens = tokenizer.encode(text).ids
    sequences = [tokens[i:i + seq_length] for i in range(0, len(tokens) - seq_length + 1, seq_length // 2)]
    return sequences

def tokenize_texts_in_parallel(texts: List[str], tokenizer: Tokenizer, seq_length: int, num_workers: int = 4) -> List[List[int]]:
    with Pool(num_workers) as pool:
        results = pool.starmap(tokenize_text, [(text, tokenizer, seq_length) for text in texts])
    return [sequence for result in results for sequence in result]  # Flatten results

def initialize_pinecone(index_name: str, api_key: str, environment: str, embedding_dimension: int):
    pinecone.init(api_key=api_key, environment=environment)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=embedding_dimension)
    index = pinecone.Index(index_name)
    return index

def prepare_datasets(texts, tokenizer, config):
    logging.info("Tokenizing datasets...")
    seq_length = config['model']['src_seq_len']
    tokenized_sequences = tokenize_texts_in_parallel(texts, tokenizer, seq_length, num_workers=4)

    logging.info("Preparing datasets...")
    total_size = len(tokenized_sequences)
    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_sequences = [tokenized_sequences[i] for i in train_indices]
    val_sequences = [tokenized_sequences[i] for i in val_indices]
    test_sequences = [tokenized_sequences[i] for i in test_indices]

    return train_sequences, val_sequences, test_sequences

def main():
    parser = argparse.ArgumentParser(description='Train transformer-based embeddings model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--local_data_dir', type=str, help='Path to local data directory')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training')

    args = parser.parse_args()

    # Initialize configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config

    # Setup logging
    setup_logging(config)

    # Ensure directories exist
    ensure_directories(config)

    # Initialize device
    device = initialize_device(config)
    logging.info(f"Using device: {device}")

    try:
        # Initialize managers
        data_manager = DataManager(config)
        model_manager = ModelManager(config, device)

        # Load data
        logging.info("Loading datasets...")
        openwebtext_data = data_manager.load_openwebtext()
        medical_data = data_manager.load_medical_datasets()
        local_data = data_manager.load_local_data(args.local_data_dir) if args.local_data_dir else []

        # Combine and prepare all datasets
        all_texts = openwebtext_data + medical_data + local_data
        train_sequences, val_sequences, test_sequences = prepare_datasets(all_texts, config['tokenizer']['load_path'], config)

        # Initialize model and tokenizer
        model, tokenizer = model_manager.initialize_model()

        # Create datasets
        train_dataset = CustomDataset(train_sequences)
        val_dataset = CustomDataset(val_sequences)
        test_dataset = CustomDataset(test_sequences)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['model']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=DataManager.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['model']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=DataManager.collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['model']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=DataManager.collate_fn
        )

        # Initialize training components
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['model']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(config, model, device)
        
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=config['model']['epochs'])

        # Load checkpoint if specified
        start_epoch = 0
        if args.checkpoint:
            model, optimizer, start_epoch, _ = model_manager.load_checkpoint(
                model, optimizer, args.checkpoint
            )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(start_epoch, config['model']['epochs']):
            logging.info(f"Starting epoch {epoch + 1}/{config['model']['epochs']}")
            
            # Training phase
            train_loss, train_accuracy, train_perplexity = trainer.train_epoch(
                train_loader, val_loader, optimizer, criterion, scheduler, epoch
            )
            
            # Validation phase
            val_loss, val_accuracy = trainer.validate(val_loader, criterion)
            
            # Update metrics
            metrics['train_loss'].append(train_loss)
            metrics['train_accuracy'].append(train_accuracy)
            metrics['train_perplexity'].append(train_perplexity)
            metrics['val_loss'].append(val_loss)
            metrics['val_accuracy'].append(val_accuracy)
            
            # Log metrics
            logging.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}, "
                f"Train Perplexity: {train_perplexity:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )
            
            # Save checkpoint
            model_manager.save_checkpoint(model, optimizer, epoch, train_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_manager.save_checkpoint(
                    model, optimizer, epoch, val_loss, is_best=True
                )
            else:
                patience_counter += 1
                if patience_counter >= config['model']['patience']:
                    logging.info("Early stopping triggered")
                    break

        # Save tokenizer after training
        model_manager.save_tokenizer(tokenizer)

        # Initialize Pinecone
        pinecone_api_key = config['pinecone']['api_key']
        pinecone_env = config['pinecone']['environment']
        index_name = config['pinecone']['index_name']
        embedding_dimension = config['model']['d_model']

        index = initialize_pinecone(index_name, pinecone_api_key, pinecone_env, embedding_dimension)

        # Generate and save embeddings
        logging.info("Generating embeddings...")
        embedding_generator = EmbeddingGenerator(model, device, index)
        embeddings = embedding_generator.generate_embeddings([data['input_ids'].tolist() for data in test_dataset])

        # Create visualizations
        logging.info("Creating visualizations...")
        visualizer = Visualizer()
        visualizer.plot_metrics(metrics, config)
        visualizer.plot_embeddings_3d(embeddings, method="PCA", config=config)
        visualizer.plot_embeddings_3d(embeddings, method="t-SNE", config=config)

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
