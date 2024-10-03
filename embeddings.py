import torch
from tokenizers import Tokenizer
from Transformer import model  # Ensure this is the correct import from your model file
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import umap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Load the BPE tokenizer
tokenizer = Tokenizer.from_file("bpe_token.json")

# Initialize the transformer model
d_model = 512  # embedding dimension
src_leq_len = 1024  # Maximum sequence length per batch
src_vocab_size = len(tokenizer.get_vocab())  # vocab size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # assuming you want the same vocab size for target

# Build transformer model with manageable sequence length
transformer_model = model.build_transformer(src_vocab_size, tgt_vocab_size, src_leq_len=src_leq_len, tgt_seq_len=src_leq_len, d_model=d_model)

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
for batch in input_ids_batches:
    # Ensure the input batch has the correct type and batch dimension
    input_ids = torch.tensor([batch], dtype=torch.long)

    # Generate embeddings from the transformer encoder for this batch
    with torch.no_grad():
        src_mask = None  # Optionally set mask for attention
        embeddings = transformer_model.encode(input_ids, src_mask)

    # Collect embeddings for this batch
    all_embeddings.append(embeddings.squeeze(0).detach())

# Concatenate all batch embeddings into a single tensor
all_embeddings_tensor = torch.cat(all_embeddings, dim=0)

# Convert the embeddings tensor to numpy for further processing
embedding_np = all_embeddings_tensor.numpy()

# ----------------------------------------------
# Option 1: Downsampled Embeddings (Choose Factor)
# ----------------------------------------------
downsample_factor = 10  # Adjust as needed
downsampled_embeddings = all_embeddings_tensor[::downsample_factor]

# Now calculate pairwise distances on the downsampled embeddings
print("Calculating distances for downsampled embeddings...")
distances = pdist(downsampled_embeddings.numpy())
distance_matrix = squareform(distances)
print("Pairwise distances for downsampled embeddings calculated.")

# ----------------------------------------------
# Option 2: PCA for Dimensionality Reduction
# ----------------------------------------------
print("Reducing dimensionality using PCA...")
pca = PCA(n_components=10)  # Reducing to 10 dimensions
reduced_embeddings = pca.fit_transform(embedding_np)

# Calculate pairwise distances on reduced embeddings
distances_pca = pdist(reduced_embeddings)
distance_matrix_pca = squareform(distances_pca)
print("Pairwise distances for PCA-reduced embeddings calculated.")

# ----------------------------------------------
# Option 3: Batch Processing for Distance Calculation with Handling Last Batch
# ----------------------------------------------
import numpy as np
from scipy.spatial.distance import pdist, squareform

def batch_pdist(embeddings, batch_size):
    num_batches = len(embeddings) // batch_size
    remainder = len(embeddings) % batch_size
    distance_matrices = []

    # Process full-size batches
    for i in range(num_batches):
        batch_embeddings = embeddings[i * batch_size:(i + 1) * batch_size]
        distances = pdist(batch_embeddings)  # Calculate pairwise distances within the batch
        distance_matrix = squareform(distances)
        distance_matrices.append(distance_matrix)

    # Process the last batch (remainder), if any
    if remainder > 0:
        last_batch_embeddings = embeddings[num_batches * batch_size:]
        distances = pdist(last_batch_embeddings)  # Calculate pairwise distances for the smaller batch
        distance_matrix = squareform(distances)
        # Pad the smaller matrix to match the shape of the larger ones
        padded_matrix = np.zeros((batch_size, batch_size))
        padded_matrix[:remainder, :remainder] = distance_matrix
        distance_matrices.append(padded_matrix)

    # Concatenate distance matrices row-wise and column-wise
    combined_matrix = np.block([[distance_matrices[i] for i in range(num_batches)]])
    
    return combined_matrix

# Example usage
batch_size = 1024  # Adjust according to your needs
distance_matrix_batch = batch_pdist(all_embeddings_tensor.numpy(), batch_size=batch_size)

print("Batch pairwise distances calculated.")

# ----------------------------------------------
# Visualization of Embeddings using UMAP or t-SNE
# ----------------------------------------------

# Apply UMAP for dimensionality reduction to 3D
print("Applying UMAP for visualization...")
umap_model = umap.UMAP(n_components=3, random_state=42)
reduced_embeddings_umap = umap_model.fit_transform(reduced_embeddings)

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D reduced embeddings
ax.scatter(reduced_embeddings_umap[:, 0], reduced_embeddings_umap[:, 1], reduced_embeddings_umap[:, 2])
ax.set_title('3D UMAP of Embeddings')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

plt.show()
