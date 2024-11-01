import streamlit as st
import matplotlib.pyplot as plt
from embeddings import (
    initialize_model,
    fine_tune_model_with_early_stopping,
    generate_embeddings,
    plot_embeddings,
    calculate_sampled_cosine_similarity,
    get_top_tokens,
    tokenize_data,
    CustomDataset,
    collate_fn,
)
import torch
from torch.utils.data import DataLoader
import numpy as np
import os

# Main UI code
def main():
    st.title("LuminaLM Training Dashboard")
    
    # Load data directory path
    data_dir = st.text_input("Enter data directory path", "")
    
    if not data_dir:
        st.error("Please provide a valid data directory")
        return
    
    # Initialize the model and tokenizer with d_model=512
    transformer_model, tokenizer = initialize_model(tokenizer_path="LuminaLM_text_token.json", d_model=512)

    # Tokenize the data
    st.write("Tokenizing data... This may take some time.")
    input_ids_batches, target_ids_batches = tokenize_data(tokenizer, data_dir)
    
    # Create DataLoader for training and validation
    dataset = CustomDataset(tokenized_inputs=input_ids_batches, tokenized_targets=target_ids_batches)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)  # Assuming you have a validation set
    
    # Fine-tune the model with early stopping and model saving logic
    st.write("Fine-tuning the model...")
    loss_values, accuracy_values, perplexity_values, val_loss_values, val_accuracy_values, embeddings = fine_tune_model_with_early_stopping(
        transformer_model, train_loader, input_ids_batches, val_loader, epochs=5, lr=5e-5, patience=3
    )

    # Plot Training Loss
    st.write("Training and Validation Loss Over Epochs")
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(loss_values, label='Training Loss', color='blue')
    ax_loss.plot(val_loss_values, label='Validation Loss', color='orange')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    st.pyplot(fig_loss)
    plt.savefig('TrainingLoss.png')  # Save the training loss plot locally

    # Plot Training Accuracy
    st.write("Training and Validation Accuracy Over Epochs")
    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(accuracy_values, label='Training Accuracy', color='blue')
    ax_acc.plot(val_accuracy_values, label='Validation Accuracy', color='orange')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    st.pyplot(fig_acc)
    plt.savefig('TrainingAccuracy.png')  # Save the accuracy plot locally

    # Plot Perplexity
    st.write("Training Perplexity Over Epochs")
    fig_perp, ax_perp = plt.subplots()
    ax_perp.plot(perplexity_values, label='Perplexity')
    ax_perp.set_xlabel('Epochs')
    ax_perp.set_ylabel('Perplexity')
    ax_perp.legend()
    st.pyplot(fig_perp)
    plt.savefig('TrainingPerplexity.png')  # Save the perplexity plot locally

    # Generate embeddings after fine-tuning
    st.write("Generating embeddings after fine-tuning...")
    st.write(f"Total Embeddings Generated: {embeddings.shape[0]}")

    # Save the model after generating embeddings (i.e., after 5th epoch)
    st.write("Model saved after generating embeddings.")

    # Visualizations: Sample for PCA, t-SNE, and Cosine Similarity
<<<<<<< HEAD
    sample_size = st.slider("Select Sample Size for Visualization", min_value=1000, max_value=500000, value=1000)
=======
    sample_size = 500000
>>>>>>> 462031e4bf56ff98ac912176fdfcea6606899f22

    # Sample embeddings for visualization
    sample_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]

    # PCA Visualization
    st.write("PCA Visualization")
    plot_embeddings(sample_embeddings.numpy(), method="PCA", sample_size=sample_size)

    # t-SNE Visualization
    st.write("t-SNE Visualization")
    plot_embeddings(sample_embeddings.numpy(), method="t-SNE", sample_size=sample_size)

    # Cosine Similarity Visualization
    st.write("Cosine Similarity Visualization")
    calculate_sampled_cosine_similarity(embeddings.numpy(), sample_size=sample_size)

    # Display Top Tokens
    st.write("Top Tokens from Tokenized Data")
    top_tokens = get_top_tokens(tokenizer, input_ids_batches)
    for token, freq in top_tokens:
        st.write(f"Token: {token}, Frequency: {freq}")
    
if __name__ == "__main__":
    main()
