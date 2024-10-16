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
from torch.utils.data import DataLoader, Dataset, default_collate
import torch.nn.utils.rnn as rnn_utils

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Initialize the transformer model with d_model=256
def initialize_model(tokenizer_path="bpe_token.json", d_model=256, src_leq_len=512):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    src_vocab_size = tokenizer.get_vocab_size()
    tgt_vocab_size = src_vocab_size
    
    # Initialize the transformer model with d_model=256
    transformer_model = model.build_transformer(
        src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model
    ).to(device)
    
    return transformer_model, tokenizer

# Custom dataset
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

# Define a collate function to pad sequences
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]

    # Pad input and target sequences to the maximum length in the batch
    input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = rnn_utils.pad_sequence(target_ids, batch_first=True, padding_value=0)

    return {"input_ids": input_ids_padded, "target_ids": target_ids_padded}

# Tokenize data
def tokenize_data(tokenizer, directory, batch_size=128):
    encoded_input = []
    encoded_target = []

    def read_files_in_chunks(directory, chunk_size=10000):
        file_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".txt")]
        for file_name in file_list:
            with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

    for chunk in read_files_in_chunks(directory):
        encoded_input.extend(tokenizer.encode(chunk).ids)
        encoded_target.extend(tokenizer.encode(chunk).ids)  # Assuming source=target setup

    input_ids_batches = [encoded_input[i:i + batch_size] for i in range(0, len(encoded_input), batch_size)]
    target_ids_batches = [encoded_target[i:i + batch_size] for i in range(0, len(encoded_target), batch_size)]

    return input_ids_batches, target_ids_batches

# Fine-tune model
def fine_tune_model(model, train_loader, epochs=3, lr=5e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            outputs = model(input_ids, target_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(train_loader)}")

def save_model(model, path="fine_tuned_transformer_model.pth"):
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    else:
        raise ValueError("The provided object is not a PyTorch model.")

# Generate embeddings post-training
def generate_embeddings(model, input_ids_batches):
    model.eval()
    all_embeddings = []
    with tqdm(total=len(input_ids_batches), desc="Generating Embeddings") as pbar_batches:
        for batch in input_ids_batches:
            input_ids = torch.tensor([batch], dtype=torch.long).to(device)
            src_mask = torch.ones(input_ids.shape).to(device)  # Create a mask if necessary, or pass None
            with torch.no_grad():
                embeddings = model.encode(input_ids, src_mask)  # Ensure the mask is passed
            all_embeddings.append(embeddings.squeeze(0).detach().cpu())
            pbar_batches.update(1)

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    return all_embeddings_tensor

transformer_model, tokenizer = initialize_model()

# After fine-tuning or training
save_model(transformer_model, "fine_tuned_transformer_model.pth")

# PCA and t-SNE plotting
def plot_embeddings(embeddings_np, method="PCA"):
    if method == "PCA":
        pca = PCA(n_components=3)
        reduced_embeddings = pca.fit_transform(embeddings_np)
        title = "3D PCA Projection"
        filename = "3d_pca_projection.png"
    elif method == "t-SNE":
        tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=300)
        reduced_embeddings = tsne.fit_transform(embeddings_np)
        title = "3D t-SNE Projection"
        filename = "3d_tsne_projection.png"
    
    # 3D Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], alpha=0.5)
    ax.set_title(title)
    plt.savefig(filename)
    plt.show()

# Calculate cosine similarity
def calculate_cosine_similarity(embeddings_np):
    max_samples = 5000
    if embeddings_np.shape[0] > max_samples:
        indices = np.random.choice(embeddings_np.shape[0], max_samples, replace=False)
        embedding_np_sampled = embeddings_np[indices]
    else:
        embedding_np_sampled = embeddings_np

    cos_sim_matrix_sampled = cosine_similarity(embedding_np_sampled)
    sns.heatmap(cos_sim_matrix_sampled, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('Cosine Similarity Matrix (Sampled)')
    plt.savefig('cosine_similarity_sampled.png')
    plt.show()