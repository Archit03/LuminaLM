import torch
import torch.nn as nn
from tokenizers import Tokenizer
from Transformer import model  # Ensure this is the correct module and function
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from tqdm import tqdm
import os
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Check if CUDA is available; otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the BPE tokenizer
tokenizer = Tokenizer.from_file("bpe_token.json")

# Initialize the transformer model
d_model = 256  # Reduced embedding dimension from 512 to 256 for less memory usage
src_leq_len = 512  # Reduced sequence length for memory management
src_vocab_size = len(tokenizer.get_vocab())  # Vocabulary size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # Assuming the same vocab size for target

# Build transformer model with smaller embedding size and sequence length, and move it to the device
transformer_model = model.build_transformer(
    src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model
).to(device)
transformer_model.eval()

# Specify the directory containing the text files
directory_path = "/content/Sentient-Sculptor-LLM/Data"  # Path to your directory

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

### PCA for 3D projection ###
with tqdm(desc="PCA Reduction", total=1) as pbar_pca:
    pca = PCA(n_components=3)  # 3 components for 3D
    reduced_embeddings_pca = pca.fit_transform(embedding_np)
    pbar_pca.update(1)

# 3D Plotting PCA projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_embeddings_pca[:, 0], reduced_embeddings_pca[:, 1], reduced_embeddings_pca[:, 2], alpha=0.5)
ax.set_title('3D PCA Projection of the Embeddings')
plt.savefig('3d_pca_projection.png')
plt.show()

### Cosine Similarity Block-wise Calculation with Memory Mapping ###
def compute_cosine_similarity_blockwise_to_disk(embeddings, block_size=1000, output_file="cosine_similarity_blockwise.npy"):
    n = embeddings.shape[0]
    
    # Memory map the file to store blocks incrementally without holding the full matrix in memory
    cos_sim_matrix_file = np.memmap(output_file, dtype='float32', mode='w+', shape=(n, n))

    for i in tqdm(range(0, n, block_size), desc="Calculating Cosine Similarity in Blocks"):
        for j in range(0, n, block_size):
            # Compute the cosine similarity for the current block
            block_sim = cosine_similarity(embeddings[i:i+block_size], embeddings[j:j+block_size])
            # Save the block to the memory-mapped file
            cos_sim_matrix_file[i:i+block_size, j:j+block_size] = block_sim
    
    # Flush the memory-mapped file to ensure data is written to disk
    cos_sim_matrix_file.flush()
    return output_file

# Use block-wise cosine similarity calculation and store results to disk
block_size = 500  # Adjust block size based on available memory
output_file = "cosine_similarity_blockwise.npy"
cos_sim_matrix_blockwise_file = compute_cosine_similarity_blockwise_to_disk(embedding_np, block_size=block_size, output_file=output_file)

# Once saved to disk, you can load parts of the matrix when needed without requiring full memory:
cos_sim_matrix_blockwise = np.memmap(output_file, dtype='float32', mode='r', shape=(len(embedding_np), len(embedding_np)))

# Optionally, plot a sample of the matrix
sample_size = 1000  # For visualization, sample a subset
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix_blockwise[:sample_size, :sample_size], cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Matrix (Block-wise, Sampled)')
plt.savefig('cosine_similarity_blockwise_sampled.png')
plt.show()

### t-SNE for 3D projection ###
from sklearn.manifold import TSNE

# Perform t-SNE with 3 components for 3D visualization
with tqdm(desc="t-SNE Reduction", total=1) as pbar_tsne:
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=300)
    reduced_embeddings_tsne = tsne.fit_transform(embedding_np)
    pbar_tsne.update(1)

# 3D Plotting t-SNE projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_embeddings_tsne[:, 0], reduced_embeddings_tsne[:, 1], reduced_embeddings_tsne[:, 2], alpha=0.5)
ax.set_title('3D t-SNE Projection of the Embeddings')
plt.savefig('3d_tsne_projection.png')
plt.show()
