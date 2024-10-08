import torch
import torch.nn as nn
from tokenizers import Tokenizer
from Transformer import model  # Ensure this module is correctly defined with build_transformer method
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import umap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from tqdm import tqdm
import numba
from concurrent.futures import ThreadPoolExecutor

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BPE tokenizer
tokenizer = Tokenizer.from_file("bpe_token.json")

# Initialize the transformer model with specified configuration
d_model = 512  # Embedding dimension
src_leq_len = 1064  # Maximum sequence length per batch
src_vocab_size = len(tokenizer.get_vocab())  # Vocabulary size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # Assuming the same vocab size for target

transformer_model = model.build_transformer(src_vocab_size, tgt_vocab_size, src_leq_len, src_leq_len, d_model).to(device)
transformer_model.eval()

# Function to process a single batch and generate embeddings
def process_batch(batch):
    input_ids = torch.tensor([batch], dtype=torch.long).to(device)
    with torch.no_grad():
        embeddings = transformer_model.encode(input_ids, None)
    return embeddings.squeeze(0).detach().cpu().numpy()

# Read and tokenize input text from files
file_list = ["ACS_CA3_Book.txt", "Genomes_3 - T.A. Brown_.txt", "input.txt", "data.txt", "train.txt", "test.txt"]
text = ""
for file_name in file_list:
    with open(file_name, "r", encoding="utf-8") as f:
        text += f.read()

encoded_input = tokenizer.encode(text)
input_ids_batches = [encoded_input.ids[i:i + src_leq_len] for i in range(0, len(encoded_input.ids), src_leq_len)]

# Use ThreadPoolExecutor to process batches in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    all_embeddings = list(tqdm(executor.map(process_batch, input_ids_batches), total=len(input_ids_batches), desc="Processing Batches"))

# Concatenate all embeddings into a single NumPy array
embedding_np = np.vstack(all_embeddings)

# Optimized Cosine Similarity Calculation using Numba
@numba.jit(nopython=True, parallel=True)
def fast_cosine_similarity(data):
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    normalized_data = data / norms
    return np.dot(normalized_data, normalized_data.T)

cos_sim_matrix = fast_cosine_similarity(embedding_np)

# Perform PCA Reduction
pca = PCA(n_components=3)
reduced_embeddings_pca = pca.fit_transform(embedding_np)

# Perform UMAP Reduction
umap_reducer = umap.UMAP(n_components=3, random_state=42)
reduced_embeddings_umap = umap_reducer.fit_transform(embedding_np)

# Plotting UMAP results
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings_umap[:, 0], reduced_embeddings_umap[:, 1], alpha=0.5)
plt.title('UMAP Projection of the Embeddings')
plt.show()

# Plotting Cosine Similarity Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Matrix')
plt.show()
