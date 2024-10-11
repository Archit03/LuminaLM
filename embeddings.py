import torch
import torch.nn as nn
from tokenizers import Tokenizer
from Transformer import model  # Ensure this is the correct module and function
from sklearn.decomposition import PCA
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from tqdm import tqdm
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BPE tokenizer
tokenizer = Tokenizer.from_file("bpe_token.json")

# Initialize the transformer model
d_model = 512  # Embedding dimension
src_leq_len = 1064  # Maximum sequence length per batch
src_vocab_size = len(tokenizer.get_vocab())  # Vocabulary size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # Assuming the same vocab size for target

# Build transformer model with manageable sequence length and move it to the GPU
transformer_model = model.build_transformer(src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model).to(device)
transformer_model.eval()

# Specify the directory containing the text files
directory_path = "/content/drive/MyDrive/Sentient-Sculptor-LLM/Data"  # Path to your directory

# Read input text from all files in the directory and concatenate them
text = ""
file_list = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".txt")]

# Initialize progress bar for reading files
with tqdm(total=len(file_list), desc="Reading Files") as pbar_files:
    for file_name in file_list:
        with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
            text += f.read()  # Concatenate the content of each file
        pbar_files.update(1)

# Tokenize the concatenated text using the BPE tokenizer
encoded_input = tokenizer.encode(text)

# Adjust the batch size to 512
batch_size = 512
input_ids_batches = [encoded_input.ids[i:i + batch_size] for i in range(0, len(encoded_input.ids), batch_size)]

# Initialize a list to store embeddings for all batches
all_embeddings = []

# Process each batch independently with a progress bar
with tqdm(total=len(input_ids_batches), desc="Processing Batches") as pbar_batches:
    for batch in input_ids_batches:
        input_ids = torch.tensor([batch], dtype=torch.long).to(device)

        # Forward pass through the transformer model
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

# Perform PCA reduction with progress bar
with tqdm(desc="PCA Reduction", total=1) as pbar_pca:
    pca = PCA(n_components=3)
    reduced_embeddings_pca = pca.fit_transform(embedding_np)
    pbar_pca.update(1)

# Perform UMAP reduction with progress bar
with tqdm(desc="UMAP Reduction", total=1) as pbar_umap:
    umap_reducer = umap.UMAP(n_components=3)
    reduced_embeddings_umap = umap_reducer.fit_transform(embedding_np)
    pbar_umap.update(1)

# Plotting UMAP projection
fig, ax = plt.subplots()
ax.scatter(reduced_embeddings_umap[:, 0], reduced_embeddings_umap[:, 1], alpha=0.5)
ax.set_title('UMAP Projection of the Embeddings')
plt.show()

# Calculate cosine similarity with a progress bar
with tqdm(desc="Calculating Cosine Similarities", total=1) as pbar_cosine:
    cos_sim_matrix = cosine_similarity(embedding_np)
    pbar_cosine.update(1)

# Plotting cosine similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Matrix')
plt.show()
