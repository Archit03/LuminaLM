import streamlit as st
import matplotlib.pyplot as plt
from embeddings import (
    initialize_model,
    fine_tune_model_with_early_stopping,
    plot_training_loss,
    plot_training_accuracy,
    plot_training_perplexity,
    plot_embeddings_3d,
    plot_cosine_similarity_matrix,
    load_openwebtext,
    load_local_data,
    tokenize_combined_data,
    CustomDataset,
    collate_fn,
)
import torch
from torch.utils.data import DataLoader

def main():
    st.title("LuminaLM Training Dashboard")

    # Input for local data directory
    data_dir = st.text_input("Enter local data directory path", "")
    if not data_dir:
        st.error("Please provide a valid local data directory")
        return

    # Initialize model and tokenizer
    transformer_model, tokenizer = initialize_model(tokenizer_path="LuminaLM_text_token.json", d_model=512)

    # Load datasets
    openwebtext_data = load_openwebtext()
    local_data = load_local_data(data_dir)

    # Tokenize and create batches
    input_ids_batches, target_ids_batches = tokenize_combined_data(tokenizer, openwebtext_data, local_data)
    dataset = CustomDataset(input_ids_batches, target_ids_batches)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Fine-tune model and collect metrics
    loss_values, accuracy_values, perplexity_values, val_loss_values, val_accuracy_values, embeddings = fine_tune_model_with_early_stopping(
        transformer_model, train_loader, input_ids_batches, val_loader
    )

    # Display Training Loss Plot
    st.subheader("Training Loss Over Epochs")
    plot_training_loss(loss_values)
    st.pyplot(plt.gcf())

    # Display Training Accuracy Plot
    st.subheader("Training Accuracy Over Epochs")
    plot_training_accuracy(accuracy_values)
    st.pyplot(plt.gcf())

    # Display Training Perplexity Plot
    st.subheader("Training Perplexity Over Epochs")
    plot_training_perplexity(perplexity_values)
    st.pyplot(plt.gcf())

    # Embedding Visualizations
    st.subheader("Embedding Visualizations - PCA")
    plot_embeddings_3d(embeddings, method="PCA")
    st.pyplot(plt.gcf())

    st.subheader("Embedding Visualizations - t-SNE")
    plot_embeddings_3d(embeddings, method="t-SNE")
    st.pyplot(plt.gcf())

    # Cosine Similarity Visualization
    st.subheader("Cosine Similarity Matrix (Sampled)")
    plot_cosine_similarity_matrix(embeddings)
    st.pyplot(plt.gcf())

if __name__ == "__main__":
    main()
