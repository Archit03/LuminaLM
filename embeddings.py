import torch
from tokenizers import Tokenizer
from Transformer import model  # Ensure this is the correct import from your model file
from sklearn.manifold import TSNE
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
input_ids_batches = [encoded_input.ids[i:i+batch_size] for i in range(0, len(encoded_input.ids), batch_size)]

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

# Convert the embeddings tensor to numpy for t-SNE
embedding_np = all_embeddings_tensor.numpy()  # Convert to numpy

# Apply t-SNE for dimensionality reduction to 3D
tsne = TSNE(n_components=3, random_state=42)
reduced_embeddings = tsne.fit_transform(embedding_np)

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D reduced embeddings
ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2])
ax.set_title('3D t-SNE of Embeddings')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

plt.show()
