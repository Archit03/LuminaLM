import streamlit as st
import matplotlib.pyplot as plt  # Dashboard will use these libraries
from embeddings import (
    initialize_model,
    fine_tune_model,
    generate_embeddings,
    plot_embeddings,
    calculate_cosine_similarity,
    get_top_tokens,
    tokenize_data,
    CustomDataset,
    collate_fn
)
import torch
from torch.utils.data import DataLoader

# Main UI code
def main():
    st.title("LuminaLM Training Dashboard")
    
    # Load data directory path
    data_dir = st.text_input("Enter data directory path", "")
    
    if not data_dir:
        st.error("Please provide a valid data directory")
        return
    
    # Initialize the model and tokenizer with d_model=256
    transformer_model, tokenizer = initialize_model(tokenizer_path="bpe_token.json", d_model=256)

    # Tokenize the data
    st.write("Tokenizing data...")
    input_ids_batches, target_ids_batches = tokenize_data(tokenizer, data_dir)
    
    # Create DataLoader
    dataset = CustomDataset(tokenized_inputs=input_ids_batches, tokenized_targets=target_ids_batches)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Fine-tune the model
    st.write("Fine-tuning the model...")
    loss_values, accuracy_values, perplexity_values = fine_tune_model(transformer_model, data_loader, epochs=3, lr=5e-5)

    # Plot Training Loss, Accuracy, and Perplexity
    st.write("Training Loss, Accuracy, and Perplexity")
    fig, ax = plt.subplots()
    ax.plot(loss_values, label='Loss')
    ax.plot(accuracy_values, label='Accuracy')
    ax.plot(perplexity_values, label='Perplexity')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)
    
    # Generate embeddings
    st.write("Generating embeddings...")
    embeddings = generate_embeddings(transformer_model, input_ids_batches)
    
    # Plot embeddings (PCA and t-SNE on a sample)
    st.write("Visualizing embeddings using PCA...")
    sample_embeddings = embeddings.numpy()[:5000]  # Sample size for PCA and t-SNE
    plot_embeddings(sample_embeddings, method="PCA")
    
    st.write("Visualizing embeddings using t-SNE...")
    plot_embeddings(sample_embeddings, method="t-SNE")
    
    # Calculate and visualize cosine similarity
    st.write("Calculating and visualizing cosine similarity...")
    calculate_cosine_similarity(sample_embeddings)
    
    # Show top tokens
    st.write("Top Tokens in the Dataset")
    top_tokens = get_top_tokens(tokenizer, input_ids_batches, top_n=10)
    for token, count in top_tokens:
        st.write(f"Token: {token}, Frequency: {count}")
    
if __name__ == "__main__":
    main()

