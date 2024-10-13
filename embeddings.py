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

# Create a directory to save the plots
output_directory = "/content/Sentient-Sculptor-LLM/Plots"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

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

### OPTION 1: Sampling Subset for Cosine Similarity ###
max_samples = 5000  # Reduce sample size for cosine similarity calculation
if embedding_np.shape[0] > max_samples:
    indices = np.random.choice(embedding_np.shape[0], max_samples, replace=False)
    embedding_np_sampled = embedding_np[indices]
else:
    embedding_np_sampled = embedding_np

# Calculate cosine similarity for the sampled embeddings
with tqdm(desc="Calculating Cosine Similarities (Sampled)", total=1) as pbar_cosine:
    cos_sim_matrix_sampled = cosine_similarity(embedding_np_sampled)
    pbar_cosine.update(1)

# Save the cosine similarity matrix (Sampled)
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix_sampled, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Matrix (Sampled)')
plt.savefig(os.path.join(output_directory, "cosine_similarity_sampled.png"))
plt.close()  # Close the plot to free memory

### OPTION 2: Block-Wise Cosine Similarity Calculation ###
def compute_cosine_similarity_blockwise(embeddings, block_size=1000):
    n = embeddings.shape[0]
    cos_sim_matrix = np.zeros((n, n))

    for i in tqdm(range(0, n, block_size), desc="Calculating Cosine Similarity in Blocks"):
        for j in range(0, n, block_size):
            cos_sim_matrix[i:i+block_size, j:j+block_size] = cosine_similarity(
                embeddings[i:i+block_size], embeddings[j:j+block_size]
            )
    return cos_sim_matrix

# Use block-wise cosine similarity calculation
block_size = 500  # Adjust block size based on available memory
cos_sim_matrix_blockwise = compute_cosine_similarity_blockwise(embedding_np_sampled, block_size=block_size)

# Save the block-wise cosine similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cos_sim_matrix_blockwise, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Matrix (Block-wise)')
plt.savefig(os.path.join(output_directory, "cosine_similarity_blockwise.png"))
plt.close()

### OPTION 3: Dimensionality Reduction ###
# Perform PCA reduction with progress bar
with tqdm(desc="PCA Reduction", total=1) as pbar_pca:
    pca = PCA(n_components=2)  # Reduced to 2 components for visualization
    reduced_embeddings_pca = pca.fit_transform(embedding_np_sampled)
    pbar_pca.update(1)

# Save the PCA projection plot
fig, ax = plt.subplots()
ax.scatter(reduced_embeddings_pca[:, 0], reduced_embeddings_pca[:, 1], alpha=0.5)
ax.set_title('PCA Projection of the Embeddings')
plt.savefig(os.path.join(output_directory, "pca_projection.png"))
plt.close()
