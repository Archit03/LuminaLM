from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import embeddings

# Convert the tensor to numpy for PCA
embedding_np = embeddings.squeeze(0).detach().numpy()

# Apply PCA for dimensionality reduction (from 512 to 2 components for visualization)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embedding_np)

# Plot the reduced embeddings
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
plt.title("2D PCA of Embeddings")
plt.show()
