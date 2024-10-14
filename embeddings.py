import torch
import torch.nn as nn
from tokenizer import Tokenizer
from Transformer import model  # Ensure this is the correct module and function
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Check if CUDA is available; otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the BPE tokenizer
tokenizer = Tokenizer.from_file("bpe_token.json")

# Initialize the transformer model
d_model = 512  # Increased embedding dimension to capture richer features
src_leq_len = 512  # Keep manageable sequence length for memory management
src_vocab_size = len(tokenizer.get_vocab())  # Vocabulary size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # Assuming the same vocab size for target

# Build transformer model with larger embedding size, move it to the device
transformer_model = model.build_transformer(
    src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model
).to(device)

# Define a custom dataset for input and target sequences
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

# Fine-tuning function with defined input_ids and target_ids
def fine_tune_model(model, train_loader, epochs=3, lr=5e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            # Extract input_ids and target_ids from the batch
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Forward pass through the model
            outputs = model(input_ids, target_ids)

            # Calculate the loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(train_loader)}")

# Tokenize the data (Example tokenization process)
directory_path = "/content/Sentient-Sculptor-LLM/Data"  # Path to your directory

# Process files one by one to reduce memory load
def read_files_in_chunks(directory, chunk_size=10000):
    file_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".txt")]
    for file_name in file_list:
        with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

# Tokenize the source (input) and target sequences
encoded_input = []
encoded_target = []  # Replace with actual target data
for chunk in read_files_in_chunks(directory_path):
    encoded_input.extend(tokenizer.encode(chunk).ids)
    encoded_target.extend(tokenizer.encode(chunk).ids)  # Assuming a simple source=target setup for demo

# Create a dataset and dataloader for training
batch_size = 512
input_ids_batches = [encoded_input[i:i + batch_size] for i in range(0, len(encoded_input), batch_size)]
target_ids_batches = [encoded_target[i:i + batch_size] for i in range(0, len(encoded_target), batch_size)]

# Custom dataset and dataloader
train_dataset = CustomDataset(tokenized_inputs=input_ids_batches, tokenized_targets=target_ids_batches)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Fine-tune the model
fine_tune_model(transformer_model, train_loader, epochs=3, lr=5e-5)

### Post-training Evaluation ###
# Once fine-tuning is done, you can use the model to generate embeddings
transformer_model.eval()
all_embeddings = []
with tqdm(total=len(input_ids_batches), desc="Generating Embeddings") as pbar_batches:
    for batch in input_ids_batches:
        input_ids = torch.tensor([batch], dtype=torch.long).to(device)
        with torch.no_grad():
            embeddings = transformer_model.encode(input_ids)
        all_embeddings.append(embeddings.squeeze(0).detach().cpu())  # Move to CPU for processing
        pbar_batches.update(1)

# Save the fine-tuned embeddings
all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
np.save('fine_tuned_embeddings.npy', all_embeddings_tensor.numpy())

### PCA for 3D projection ###
with tqdm(desc="PCA Reduction", total=1) as pbar_pca:
    pca = PCA(n_components=3)  # 3 components for 3D
    reduced_embeddings_pca = pca.fit_transform(all_embeddings_tensor.numpy())
    pbar_pca.update(1)

# Save the reduced PCA embeddings
pca_df = pd.DataFrame(reduced_embeddings_pca, columns=['Component 1', 'Component 2', 'Component 3'])
pca_df.to_csv('reduced_pca_embeddings.csv', index=False)

# 3D Plotting PCA projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_embeddings_pca[:, 0], reduced_embeddings_pca[:, 1], reduced_embeddings_pca[:, 2], alpha=0.5)
ax.set_title('3D PCA Projection of the Embeddings')
plt.savefig('3d_pca_projection.png')
plt.show()

### Sampling for Cosine Similarity Calculation ###
max_samples = 5000  # Limit the number of samples for cosine similarity calculation
if all_embeddings_tensor.shape[0] > max_samples:
    indices = np.random.choice(all_embeddings_tensor.shape[0], max_samples, replace=False)
    embedding_np_sampled = all_embeddings_tensor.numpy()[indices]
else:
    embedding_np_sampled = all_embeddings_tensor.numpy()

# Calculate cosine similarity for the sampled embeddings
with tqdm(desc="Calculating Cosine Similarities (Sampled)", total=1) as pbar_cosine:
    cos_sim_matrix_sampled = cosine_similarity(embedding_np_sampled)
    pbar_cosine.update(1)

# Save the cosine similarity matrix
np.save('cosine_similarity_sampled.npy', cos_sim_matrix_sampled)

# Save the cosine similarity heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix_sampled, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Matrix (Sampled)')
plt.savefig('cosine_similarity_sampled.png')
plt.show()

### t-SNE for 3D projection ###
with tqdm(desc="t-SNE Reduction", total=1) as pbar_tsne:
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=300)
    reduced_embeddings_tsne = tsne.fit_transform(embedding_np_sampled)
    pbar_tsne.update(1)

# Save the t-SNE reduced embeddings
tsne_df = pd.DataFrame(reduced_embeddings_tsne, columns=['Component 1', 'Component 2', 'Component 3'])
tsne_df.to_csv('reduced_tsne_embeddings.csv', index=False)

# 3D Plotting t-SNE projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_embeddings_tsne[:, 0], reduced_embeddings_tsne[:, 1], reduced_embeddings_tsne[:, 2], alpha=0.5)
ax.set_title('3D t-SNE Projection of the Embeddings')
plt.savefig('3d_tsne_projection.png')
plt.show()
