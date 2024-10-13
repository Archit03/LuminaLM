import torch
import torch.nn as nn
import tokenizer
from Transformer import model  # Ensure this is the correct module and function
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from tqdm import tqdm
import os
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import pandas as pd

# Check if CUDA is available; otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the BPE tokenizer (Ensure tokenizer is domain-specific if possible)
tokenizer = tokenizer.from_file("bpe_token.json")

# Initialize the transformer model (fine-tune this on domain-specific data)
d_model = 512  # Increased embedding dimension to capture richer features
src_leq_len = 512  # Keep manageable sequence length for memory management
src_vocab_size = len(tokenizer.get_vocab())  # Vocabulary size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # Assuming the same vocab size for target

# Build transformer model with larger embedding size, move it to the device
transformer_model = model.build_transformer(
    src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model
).to(device)

# Optionally fine-tune the model with your domain-specific data
def fine_tune_model(train_data, model, epochs=3, lr=1e-5):
    # Define a simple training loop
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}")

# Data preprocessing: increase data quality through augmentation or cleaning
directory_path = "/content/Sentient-Sculptor-LLM/Data"

# Process files one by one to reduce memory load
def read_files_in_chunks(directory, chunk_size=10000):
    text = ""
    file_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".txt")]
    for file_name in file_list:
        with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

# Tokenize the concatenated text in chunks using the BPE tokenizer
encoded_input = []
for chunk in read_files_in_chunks(directory_path):
    encoded_input.extend(tokenizer.encode(chunk).ids)

# Adjust the batch size for memory efficiency
batch_size = 512  # Increased batch size if memory allows
input_ids_batches = [encoded_input[i:i + batch_size] for i in range(0, len(encoded_input), batch_size)]

# Initialize a list to store embeddings for all batches
all_embeddings = []

# Process each batch independently with a progress bar
with tqdm(total=len(input_ids_batches), desc="Processing Batches") as pbar_batches:
    for batch in input_ids_batches:
        input_ids = torch.tensor([batch], dtype=torch.long).to(device)

        # Forward pass through the transformer model with dropout regularization
        transformer_model.eval()
        with torch.no_grad():
            src_mask = None  # Optional mask
            embeddings = transformer_model.encode(input_ids, src_mask)
        
        # Collect embeddings for this batch
        all_embeddings.append(embeddings.squeeze(0).detach().cpu())  # Move the embeddings to CPU for further processing
        pbar_batches.update(1)

# Concatenate all batch embeddings into a single tensor
all_embeddings_tensor = torch.cat(all_embeddings, dim=0)

# Convert the embeddings tensor to numpy for further processing
embedding_np = all_embeddings_tensor.numpy()

### PCA for 3D projection ###
with tqdm(desc="PCA Reduction", total=1) as pbar_pca:
    pca = PCA(n_components=3)  # 3 components for 3D
    reduced_embeddings_pca = pca.fit_transform(embedding_np)
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
if embedding_np.shape[0] > max_samples:
    indices = np.random.choice(embedding_np.shape[0], max_samples, replace=False)
    embedding_np_sampled = embedding_np[indices]
else:
    embedding_np_sampled = embedding_np

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
