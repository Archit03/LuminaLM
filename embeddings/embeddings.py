import torch
import torch.nn as nn
import torch.distributed as dist
from tokenizers import Tokenizer
from Transformer import model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from tqdm import tqdm
import os
import gc
import json
import logging
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.utils.rnn as rnn_utils
from pineconedb import save_embeddings_to_pinecone
from datasets import load_dataset
from typing import List, Dict, Any, Optional, Tuple
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
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
                'warmup_steps': 1000
            }
        }

class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def load_openwebtext(self) -> List[str]:
        try:
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
        datasets_to_load = ["pubmed_qa", "mednli", "i2b2_2010", "mimic_notes", "scicite"]
        texts = []
        for dataset_name in datasets_to_load:
            try:
                dataset = load_dataset(dataset_name, split="train", trust_remote_code=False)
                if "text" in dataset.column_names:
                    texts.extend(dataset["text"])
                elif all(col in dataset.column_names for col in ["question", "context"]):
                    texts.extend([f"{q.strip()} {c.strip()}" 
                                for q, c in zip(dataset["question"], dataset["context"])])
                elif "sentence" in dataset.column_names:
                    texts.extend(dataset["sentence"])
            except Exception as e:
                logging.warning(f"Error loading dataset {dataset_name}: {str(e)}")
                continue
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
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        target_ids = [torch.tensor(item['target_ids'], dtype=torch.long) for item in batch] if 'target_ids' in batch[0] else None

        input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)

        if target_ids:
            target_ids_padded = rnn_utils.pad_sequence(target_ids, batch_first=True, padding_value=0)
            return {"input_ids": input_ids_padded, "target_ids": target_ids_padded}

        return {"input_ids": input_ids_padded}

class CustomDataset(Dataset):
    def __init__(self, tokenized_inputs: List[int], tokenized_targets: Optional[List[int]] = None):
        self.inputs = tokenized_inputs
        self.targets = tokenized_targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.inputs[idx]
        if self.targets is not None:
            target_ids = self.targets[idx]
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "target_ids": torch.tensor(target_ids, dtype=torch.long)
            }
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}
class ModelManager:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def initialize_model(self, tokenizer_path: str) -> Tuple[nn.Module, Tokenizer]:
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
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
                       epoch: int, loss: float) -> None:
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

class Trainer:
    def __init__(self, config: Dict[str, Any], model: nn.Module, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.scaler = torch.amp.GradScaler()

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float, float]:
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
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                self.scaler.step(optimizer)
                self.scaler.update()

            total_loss += loss.item()
            total_perplexity += perplexity.item()
            _, predicted = torch.max(outputs, -1)
            correct_predictions += (predicted == target_ids).sum().item()
            total_predictions += target_ids.numel()

            # Memory management
            del outputs, loss, scaled_loss
            torch.cuda.empty_cache()
            if batch_idx % 100 == 0:
                gc.collect()

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
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate_embeddings(self, input_ids_batches: List[List[int]], 
                          index_name: str = "luminalm-embeddings",
                          batch_size: int = 32) -> torch.Tensor:
        self.model.eval()
        all_embeddings = []
        
        for i in tqdm(range(0, len(input_ids_batches), batch_size), desc="Generating Embeddings"):
            batch = input_ids_batches[i:i + batch_size]
            input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
            
            embeddings = self.model.encode(input_ids).cpu()
            all_embeddings.extend(embeddings)
            
            batch_ids = [f"embedding_{i}_{j}" for j in range(len(embeddings))]
            try:
                save_embeddings_to_pinecone(embeddings, batch_ids, index_name=index_name)
            except Exception as e:
                logging.error(f"Error saving embeddings to Pinecone: {str(e)}")
            
            del embeddings, input_ids
            torch.cuda.empty_cache()
            
            if i % 100 == 0:
                gc.collect()
        
        return torch.cat(all_embeddings, dim=0)

class Visualizer:
    @staticmethod
    def plot_metrics(metrics: Dict[str, List[float]], save_dir: str = "plots"):
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
    def plot_embeddings_3d(embeddings: torch.Tensor, method: str = "PCA",
                          save_dir: str = "plots"):
        os.makedirs(save_dir, exist_ok=True)
        sample_size = min(len(embeddings), 1000)
        embeddings_sample = embeddings[:sample_size]
        
        try:
            if method == "PCA":
                reducer = PCA(n_components=3)
            elif method == "t-SNE":
                reducer = TSNE(n_components=3, random_state=42)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
                
            reduced_embeddings = reducer.fit_transform(embeddings_sample)
            
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
            plt.savefig(os.path.join(save_dir, f"{method}_embeddings.png"))
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting embeddings: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Train transformer model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--C:\Users\ASUS\Desktop\LuminaLM\Data', type=str, help='Path to local data directory')
    parser.add_argument('--LuminaLM_text_tokens.json', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training')
    args = parser.parse_args()

    # Initialize configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config

    # Set up distributed training if available
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Initialize model and tokenizer
        model, tokenizer = model_manager.initialize_model(args.tokenizer_path)

        # Prepare datasets
        def prepare_datasets(texts, tokenizer, config):
            # Tokenize all texts
            logging.info("Tokenizing datasets...")
            tokenized_data = []
            for text in tqdm(texts, desc="Tokenizing"):
                tokens = tokenizer.encode(text).ids
                # Create sequences of fixed length with overlap
                seq_length = config['model']['src_seq_len']
                for i in range(0, len(tokens) - seq_length + 1, seq_length // 2):
                    sequence = tokens[i:i + seq_length]
                    if len(sequence) == seq_length:
                        tokenized_data.append(sequence)

            # Create dataset splits
            total_size = len(tokenized_data)
            train_size = int(config['data']['train_split'] * total_size)
            val_size = int(config['data']['val_split'] * total_size)
            test_size = total_size - train_size - val_size

            train_data, val_data, test_data = random_split(
                tokenized_data, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

            return train_data, val_data, test_data

        # Combine and prepare all datasets
        all_texts = openwebtext_data + medical_data + local_data
        train_data, val_data, test_data = prepare_datasets(all_texts, tokenizer, config)

        # Create data loaders
        train_dataset = CustomDataset(train_data, train_data)
        val_dataset = CustomDataset(val_data, val_data)
        test_dataset = CustomDataset(test_data, test_data)

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
            weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(config, model, device)
        
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
                train_loader, optimizer, criterion
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

        # Generate and visualize embeddings
        logging.info("Generating embeddings...")
        embedding_generator = EmbeddingGenerator(model, device)
        embeddings = embedding_generator.generate_embeddings(test_data)

        # Create visualizations
        logging.info("Creating visualizations...")
        visualizer = Visualizer()
        visualizer.plot_metrics(metrics)
        visualizer.plot_embeddings_3d(embeddings, method="PCA")
        visualizer.plot_embeddings_3d(embeddings, method="t-SNE")

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()