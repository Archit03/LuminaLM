import torch
from tokenizers import Tokenizer
from Transformer import model  # Ensure this is the correct import from your model file
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import umap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BPE tokenizer
tokenizer = Tokenizer.from_file("bpe_token.json")

# Initialize the transformer model
d_model = 512  # embedding dimension
src_leq_len = 1064  # Maximum sequence length per batch
src_vocab_size = len(tokenizer.get_vocab())  # vocab size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # assuming you want the same vocab size for target

# Build transformer model with manageable sequence length and move it to the GPU
transformer_model = model.build_transformer(src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model).to(device)

# Set model to evaluation mode (no gradient tracking needed for inference)
transformer_model.eval()

# Read input text from a file
with open("ACS_CA3_Book.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the entire input text using the BPE tokenizer
encoded_input = tokenizer.encode(text)

# Split the tokenized input into batches
batch_size = 1024  # Define the batch size (equal to src_leq_len)
input_ids_batches = [encoded_input.ids[i:i + batch_size] for i in range(0, len(encoded_input.ids), batch_size)]

# Initialize a list to store embeddings for all batches
all_embeddings = []

# Process each batch independently
for batch in tqdm(input_ids_batches, desc="Processing Batches"):
    # Ensure the input batch has the correct type and batch dimension and move it to the GPU
    input_ids = torch.tensor([batch], dtype=torch.long).to(device)

    # Generate embeddings from the transformer encoder for this batch
    with torch.no_grad():
        src_mask = None  # Optionally set mask for attention
        embeddings = transformer_model.encode(input_ids, src_mask)

    # Collect embeddings for this batch
    all_embeddings.append(embeddings.squeeze(0).detach().cpu())  # Move the embeddings to CPU for further processing

# Concatenate all batch embeddings into a single tensor
all_embeddings_tensor = torch.cat(all_embeddings, dim=0)

# Convert the embeddings tensor to numpy for further processing
embedding_np = all_embeddings_tensor.numpy()

# Print all embeddings
print("All Embeddings:\n", embedding_np)

# ----------------------------------------------
# Option 1: Downsampled Embeddings (Choose Factor)
# ----------------------------------------------
downsample_factor = 10  # Adjust as needed
downsampled_embeddings = all_embeddings_tensor[::downsample_factor]

# Now calculate pairwise distances on the downsampled embeddings
print("Calculating distances for downsampled embeddings...")
distances = pdist(downsampled_embeddings.numpy())
distance_matrix = squareform(distances)
print("Pairwise distances for downsampled embeddings calculated.", distance_matrix)

# ----------------------------------------------
# Option 2: PCA for Dimensionality Reduction
# ----------------------------------------------
print("Reducing dimensionality using PCA...")
pca = PCA(n_components=10)  # Reducing to 10 dimensions
reduced_embeddings = pca.fit_transform(embedding_np)
print(reduced_embeddings)

# Calculate pairwise distances on reduced embeddings
distances_pca = pdist(reduced_embeddings)
distance_matrix_pca = squareform(distances_pca)
print("Pairwise distances for PCA-reduced embeddings calculated.", distance_matrix_pca)

# ----------------------------------------------
# Visualization of Embeddings using UMAP or t-SNE
# ----------------------------------------------
print("Applying UMAP for visualization...")
umap_model = umap.UMAP(n_components=3, random_state=42)
reduced_embeddings_umap = umap_model.fit_transform(reduced_embeddings)
print(reduced_embeddings_umap)

# 3D Plotting with larger figure size and higher resolution
fig = plt.figure(figsize=(14, 10))  # Set a larger figure size
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_embeddings_umap[:, 0], reduced_embeddings_umap[:, 1], reduced_embeddings_umap[:, 2])

# Set detailed labels and title
ax.set_title('3D UMAP of Embeddings', fontsize=20)
ax.set_xlabel('Component 1', fontsize=14)
ax.set_ylabel('Component 2', fontsize=14)
ax.set_zlabel('Component 3', fontsize=14)

# Save the 3D UMAP plot with very high resolution (dpi=3000)
plt.savefig('umap_3d_embeddings_high_res.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------
# Cosine Similarity Calculation
# ----------------------------------------------
print("Calculating cosine similarities...")

# Calculate the cosine similarity between embeddings
cos_sim_matrix = cosine_similarity(embedding_np)

# Cosine Similarity Heatmap Visualization with larger figure size for better clarity
plt.figure(figsize=(14, 10))  # Set a larger figure size
sns.heatmap(cos_sim_matrix[:50, :50], cmap='coolwarm', annot=False, cbar_kws={'shrink': 0.8})

# Add title and save the heatmap with higher resolution
plt.title('Cosine Similarity Heatmap of Embeddings', fontsize=20)
plt.savefig('cosine_similarity_heatmap_high_res.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------
# Identify the top 5 most similar pairs of embeddings based on cosine similarity
# ----------------------------------------------
def find_top_similar_pairs(cos_sim_matrix, num_top_pairs=5):
    # Get the indices of the matrix but ignore self-similarity (i.e., where indices are the same)
    num_embeddings = cos_sim_matrix.shape[0]
    
    # Create a mask to ignore diagonal elements (self-similarity)
    mask = np.ones(cos_sim_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    # Extract the valid similarities (ignoring diagonal elements)
    filtered_similarities = cos_sim_matrix[mask]

    # Get the top N most similar pairs from the filtered similarities
    sorted_indices = np.argsort(-filtered_similarities)[:num_top_pairs]

    # Recreate indices from the flattened upper triangle part of the similarity matrix
    row_indices, col_indices = np.triu_indices(num_embeddings, k=1)
    
    top_pairs = []
    for i in sorted_indices:
        r, c = row_indices[i], col_indices[i]
        similarity = cos_sim_matrix[r, c]
        top_pairs.append((r, c, similarity))

    return top_pairs

# Call the function to get the top 5 pairs
top_similar_pairs = find_top_similar_pairs(cos_sim_matrix, num_top_pairs=5)

# Print the top similar pairs
print(f"Top {len(top_similar_pairs)} most similar pairs of embeddings (index-based):")
for idx, (i, j, sim) in enumerate(top_similar_pairs):
    print(f'Pair {idx+1}: Embedding {i} and Embedding {j} with similarity {sim:.4f}')
