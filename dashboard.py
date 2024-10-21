import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from embeddings import (
    initialize_model,
    fine_tune_model,
    generate_embeddings,
    plot_embeddings,
    calculate_cosine_similarity,
    tokenize_data,
    CustomDataset,
    collate_fn,
    save_model,
    get_top_tokens
)
from torch.utils.data import DataLoader

# Main UI code
def main():
    st.title("LuminaLM Training Dashboard")
    
    # Load data directory path
    data_dir = st.text_input("Enter data directory path", "")
    
    if not data_dir:
        st.error("Please provide a valid data directory")
        return

    # Initialize the model and tokenizer
    transformer_model, tokenizer = initialize_model(tokenizer_path="bpe_token.json", d_model=256)
    
    # Tokenize the data
    st.write("Tokenizing data...")
    input_ids_batches, target_ids_batches = tokenize_data(tokenizer, data_dir)
    
    # Show top tokens
    st.write("Top Tokens:")
    top_tokens = get_top_tokens(tokenizer, input_ids_batches, top_n=10)
    for token, freq in top_tokens:
        st.write(f"Token: {token}, Frequency: {freq}")
    
    # Create DataLoader
    dataset = CustomDataset(tokenized_inputs=input_ids_batches, tokenized_targets=target_ids_batches)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Fine-tune the model
    st.write("Fine-tuning the model...")
    loss_values, accuracy_values, perplexity_values = fine_tune_model(transformer_model, data_loader, epochs=3, lr=5e-5)
    
    # Plot Loss, Accuracy, and Perplexity
    st.write("Training Loss, Accuracy, and Perplexity")
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[0].plot(loss_values, label='Loss')
    ax[1].plot(accuracy_values, label='Accuracy')
    ax[2].plot(perplexity_values, label='Perplexity')

    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[2].set_xlabel('Epochs')

    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Accuracy')
    ax[2].set_ylabel('Perplexity')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    st.pyplot(fig)
    
    # Generate embeddings
    st.write("Generating embeddings...")
    embeddings = generate_embeddings(transformer_model, input_ids_batches)
    
    # Plot PCA and t-SNE embeddings
    st.write("Visualizing embeddings using PCA...")
    plot_embeddings(embeddings.numpy(), method="PCA")
    
    st.write("Visualizing embeddings using t-SNE...")
    plot_embeddings(embeddings.numpy(), method="t-SNE")
    
    # Calculate and visualize cosine similarity
    st.write("Calculating and visualizing cosine similarity...")
    calculate_cosine_similarity(embeddings.numpy())
    
    # Save the model after all steps
    save_model(transformer_model, "fine_tuned_transformer_model.pth")

if __name__ == "__main__":
    main()
