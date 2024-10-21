import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from embeddings import (
    initialize_model,
    fine_tune_model,
    generate_embeddings,
    save_model,
    tokenize_data,
    CustomDataset,
    collate_fn,
    DataLoader
)

# Main UI code
def main():
    st.title("LuminaLM Dashboard")

    # Load data directory path
    data_dir = st.text_input("Enter data directory path", "")

    if not data_dir:
        st.error("Please provide a valid data directory")
        return

    # Initialize the model and tokenizer with d_model=256
    transformer_model, tokenizer = initialize_model(tokenizer_path="bpe_token.json", d_model=256)

    # Tokenize the data and get top tokens
    st.write("Tokenizing data...")
    input_ids_batches, target_ids_batches, top_tokens = tokenize_data(tokenizer, data_dir)

    # Display Top Tokens
    st.write("Top Tokens:")
    for token, count in top_tokens:
        st.write(f"{token}: {count}")

    # Prepare data loader
    dataset = CustomDataset(tokenized_inputs=input_ids_batches, tokenized_targets=target_ids_batches)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Split into training and validation sets
    train_loader = data_loader
    val_loader = data_loader  # Assuming you're using the same for validation for simplicity

    # Fine-tune the model
    st.write("Fine-tuning the model...")
    loss_values, accuracy_values, perplexity_values, val_loss_values = fine_tune_model(transformer_model, train_loader, val_loader, epochs=3, lr=5e-5)

    # Plot Loss, Accuracy, and Perplexity
    st.write("Training Loss, Validation Loss, Accuracy, and Perplexity")

    fig, ax = plt.subplots()
    ax.plot(loss_values, label='Training Loss')
    ax.plot(val_loss_values, label='Validation Loss')
    ax.plot(accuracy_values, label='Accuracy')
    ax.plot(perplexity_values, label='Perplexity')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

    # Generate embeddings
    st.write("Generating embeddings...")
    embeddings = generate_embeddings(transformer_model, input_ids_batches)

    # Show total count of embeddings
    total_embeddings = embeddings.shape[0]
    st.write(f"Total number of embeddings: {total_embeddings}")

    # Plot PCA on a sample of embeddings
    st.write("Visualizing PCA on embeddings sample...")
    pca = PCA(n_components=3)
    sample_size = 500  # Limit sample size for visualization
    sample_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    pca_embeddings = pca.fit_transform(embeddings[sample_indices].numpy())

    fig_pca = plt.figure()
    ax_pca = fig_pca.add_subplot(111, projection='3d')
    ax_pca.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], pca_embeddings[:, 2], alpha=0.5)
    st.pyplot(fig_pca)

    # Plot t-SNE on a sample of embeddings
    st.write("Visualizing t-SNE on embeddings sample...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=300)
    tsne_embeddings = tsne.fit_transform(embeddings[sample_indices].numpy())

    fig_tsne = plt.figure()
    ax_tsne = fig_tsne.add_subplot(111, projection='3d')
    ax_tsne.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:, 2], alpha=0.5)
    st.pyplot(fig_tsne)

    # Save model
    save_model(transformer_model)

if __name__ == "__main__":
    main()
