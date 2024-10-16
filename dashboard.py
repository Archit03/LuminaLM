import streamlit as st
from embeddings import (
    initialize_model,   # Initializes the model and tokenizer
    fine_tune_model,    # Fine-tunes the model
    generate_embeddings,  # Generates embeddings
    plot_embeddings,    # Plot PCA or t-SNE embeddings
    calculate_cosine_similarity,  # Calculate and plot cosine similarity
    tokenize_data,       # Tokenize data
    CustomDataset, 
    collate_fn,
    save_model           # Save the model at the end
)
import matplotlib.pyplot as plt
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
    loss_values, accuracy_values = fine_tune_model(transformer_model, data_loader, epochs=3, lr=5e-5)

    # Plot Loss and Accuracy
    st.write("Training Loss and Accuracy")
    fig, ax = plt.subplots()
    ax.plot(loss_values, label='Loss')
    ax.plot(accuracy_values, label='Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)
    
    # Generate embeddings
    st.write("Generating embeddings...")
    embeddings = generate_embeddings(transformer_model, input_ids_batches)
    
    # Plot embeddings
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
