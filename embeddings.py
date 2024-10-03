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
src_vocab_size = len(tokenizer.get_vocab())  # vocab size from the BPE tokenizer
tgt_vocab_size = src_vocab_size  # assuming you want the same vocab size for target

# Build transformer model
transformer_model = model.build_transformer(src_vocab_size, tgt_vocab_size, src_leq_len=1024, tgt_seq_len=1024, d_model=d_model)

# Set model to evaluation mode (no gradient tracking needed for inference)
transformer_model.eval()

# Read input text from a file
text = "This is a big deal! We have our own embeddings generator. Now, lets kick some ass at OpenAI."
# Tokenize the input using the BPE tokenizer
encoded_input = tokenizer.encode(text)

# Ensure input_ids has the correct type (Long) and batch dimension
input_ids = torch.tensor([encoded_input.ids], dtype=torch.long)  # Convert to LongTensor

# Generate embeddings from the transformer encoder
with torch.no_grad():
    src_mask = None  # Optionally set mask for attention
    embeddings = transformer_model.encode(input_ids, src_mask)
    print(embeddings)

# Convert the embeddings tensor to numpy for t-SNE
embedding_np = embeddings.squeeze(0).detach().numpy()  # Remove batch dimension and convert to numpy

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
