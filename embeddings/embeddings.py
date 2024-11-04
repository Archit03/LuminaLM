import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import logging
from pineconedb import save_embeddings_to_pinecone
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def load_openwebtext():
    dataset = load_dataset("openwebtext",  split="train[:10%]", trust_remote_code=True)
    texts = [item['text'] for item in dataset]
    return texts

def load_local_data(directory):
    texts = []
    file_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".txt")]
    for file_name in file_list:
        with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
            texts.extend(f.readlines())
    return texts

def initialize_model(tokenizer_path="LuminaLM_text_token.json", d_model=512, src_seq_len=512):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    src_vocab_size = tokenizer.get_vocab_size()
    tgt_vocab_size = src_vocab_size
    
    transformer_model = model.build_transformer(
        src_vocab_size, tgt_vocab_size, src_seq_len=src_seq_len, tgt_seq_len=src_seq_len, d_model=d_model
    ).to(device)
    
    return transformer_model, tokenizer

class CustomDataset(Dataset):
    def __init__(self, tokenized_inputs, tokenized_targets=None):
        self.inputs = tokenized_inputs
        self.targets = tokenized_targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        if self.targets is not None:
            target_ids = self.targets[idx]
            return {"input_ids": torch.tensor(input_ids, dtype=torch.long), 
                    "target_ids": torch.tensor(target_ids, dtype=torch.long)}
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = rnn_utils.pad_sequence(target_ids, batch_first=True, padding_value=0)
    return {"input_ids": input_ids_padded, "target_ids": target_ids_padded}

def tokenize_combined_data(tokenizer, openwebtext_data, local_data, batch_size=128):
    texts = openwebtext_data + local_data
    encoded_input = []
    encoded_target = []
    for text in tqdm(texts, desc="Tokenizing Combined Dataset"):
        encoded_input.extend(tokenizer.encode(text).ids)
        encoded_target.extend(tokenizer.encode(text).ids)
    input_ids_batches = [encoded_input[i:i + batch_size] for i in range(0, len(encoded_input), batch_size)]
    target_ids_batches = [encoded_target[i:i + batch_size] for i in range(0, len(encoded_target), batch_size)]
    return input_ids_batches, target_ids_batches

def fine_tune_model_with_early_stopping(model, train_loader, input_ids_batches, val_loader, epochs=5, lr=5e-5, patience=3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    loss_values, accuracy_values, perplexity_values, val_loss_values, val_accuracy_values = [], [], [], [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        total_perplexity = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(input_ids, target_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                perplexity = torch.exp(loss)
                total_perplexity += perplexity.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, -1)
            correct_predictions += (predicted == target_ids).sum().item()
            total_predictions += target_ids.numel()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        avg_perplexity = total_perplexity / len(train_loader)
        loss_values.append(avg_loss)
        accuracy_values.append(accuracy)
        perplexity_values.append(avg_perplexity)

        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        val_loss_values.append(val_loss)
        val_accuracy_values.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logging.info(f"Validation loss improved to: {val_loss:.4f}")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    embeddings = generate_embeddings(model, input_ids_batches)
    return loss_values, accuracy_values, perplexity_values, val_loss_values, val_accuracy_values, embeddings

def validate_model(model, val_loader, criterion):
    model.eval()
    total_val_loss = 0
    correct_val_predictions = 0
    total_val_predictions = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            outputs = model(input_ids, target_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, -1)
            correct_val_predictions += (predicted == target_ids).sum().item()
            total_val_predictions += target_ids.numel()
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val_predictions / total_val_predictions
    return avg_val_loss, val_accuracy

def generate_embeddings(model, input_ids_batches, index_name="luminalm-embeddings"):
    model.eval()
    all_embeddings = []
    batch_ids = []
    with tqdm(total=len(input_ids_batches), desc="Generating and Saving Embeddings") as pbar:
        for i, batch in enumerate(input_ids_batches):
            input_ids = torch.tensor([batch], dtype=torch.long).to(device)
            with torch.no_grad():
                embeddings = model.encode(input_ids).cpu()
            all_embeddings.extend(embeddings)
            
            # Prepare for saving in Pinecone
            batch_ids = [f"embedding_{i}_{j}" for j in range(len(embeddings))]
            save_embeddings_to_pinecone(embeddings, batch_ids, index_name=index_name)
            
            pbar.update(1)
    
    # Return all embeddings as a single tensor if needed locally
    return torch.cat(all_embeddings, dim=0)


# Graphing functions
def plot_training_loss(loss_values):
    plt.figure(figsize=(6, 5))
    plt.plot(loss_values, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('TrainingLoss.png')
    

def plot_training_accuracy(accuracy_values):
    plt.figure(figsize=(6, 5))
    plt.plot(accuracy_values, label='Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('TrainingAccuracy.png')

def plot_training_perplexity(perplexity_values):
    plt.figure(figsize=(6, 5))
    plt.plot(perplexity_values, label='Perplexity', color='green')
    plt.title('Training Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('TrainingPerplexity.png')

def plot_embeddings_3d(embeddings, method="PCA"):
    sample_size = min(len(embeddings), 1000)
    embeddings_sample = embeddings[:sample_size]
    if method == "PCA":
        reducer = PCA(n_components=3)
    elif method == "t-SNE":
        reducer = TSNE(n_components=3, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings_sample)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], alpha=0.5)
    ax.set_title(f'3D {method} Projection')
    plt.savefig(f"{method}.png")

def plot_cosine_similarity_matrix(embeddings, sample_size=500):
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sampled_embeddings = embeddings[sample_indices]
    cos_sim_matrix = cosine_similarity(sampled_embeddings)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_sim_matrix, cmap='viridis')
    plt.title('Cosine Similarity Matrix')
    plt.savefig('Cosine_Similarity_Matrix.png')
