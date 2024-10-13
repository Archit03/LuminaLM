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

# Check if CUDA is available; otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BPE tokenizer
tokenizer = Tokenizer.from_file("bpe_token.json")

# Initialize two transformer models with smaller embedding size
d_model = 256  # Each model will have a 256-dimensional embedding
src_leq_len = 512  # Reduced sequence length for memory management
src_vocab_size = len(tokenizer.get_vocab())  # Vocabulary size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # Assuming the same vocab size for target

# Build two transformer models
transformer_model_1 = model.build_transformer(
    src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model
).to(device)

transformer_model_2 = model.build_transformer(
    src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model
).to(device)

# Set models to evaluation mode (no gradient tracking needed for inference)
transformer_model_1.eval()
transformer_model_2.eval()

# Specify the directory containing the text files
directory_path = "C:\\Users\\LENOVO\\Desktop\\Sentient-Sculptor-LLM\\Data"  # Path to your directory

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

# Adjust the batch size to 256 for memory efficiency
batch_size = 256
input_ids_batches = [encoded_input[i:i + batch_size] for i in range(0, len(encoded_input), batch_size)]

# Initialize a list to store embeddings for all batches
all_embeddings = []

# Process each batch independently with a progress bar
with tqdm(total=len(input_ids_batches), desc="Processing Batches") as pbar_batches:
    for batch in input_ids_batches:
        input_ids = torch.tensor([batch], dtype=torch.long).to(device)

        # Forward pass through the first transformer model
        with torch.no_grad():
            src_mask = None  # Optional mask
            embeddings_1 = transformer_model_1.encode(input_ids, src_mask)
        
        # Forward pass through the second transformer model
        with torch.no_grad():
            embeddings_2 = transformer_model_2.encode(input_ids, src_mask)
        
        # Concatenate the embeddings from both models to create a 512-dimensional embedding
        combined_embeddings = torch.cat((embeddings_1, embeddings_2), dim=-1)
        
        # Collect embeddings for this batch
        all_embeddings.append(combined_embeddings.squeeze(0).detach().cpu())  # Move to CPU for further processing
        pbar_batches.update(1)

# Concatenate all batch embeddings into a single tensor
all_embeddings_tensor = torch.cat(all_embeddings, dim=0)

# Convert the embeddings tensor to numpy for further processing
embedding_np = all_embeddings_tensor.numpy()

# Perform PCA reduction with progress bar
with tqdm(desc="PCA Reduction", total=1) as pbar_pca:
    pca = PCA(n_components=2)  # Reduced to 2 components for visualization
    reduced_embeddings_pca = pca.fit_transform(embedding_np)
    pbar_pca.update(1)

# Optionally, perform UMAP reduction if memory permits (use smaller `n_neighbors` and `min_dist`)
with tqdm(desc="UMAP Reduction", total=1) as pbar_umap:
    umap_reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.3)  # Optimized UMAP parameters
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

