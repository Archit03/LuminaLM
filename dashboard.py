import streamlit as st
import matplotlib.pyplot as plt  # Dashboard will use these libraries
from embeddings import (
    initialize_model,
    fine_tune_model,
    generate_embeddings,
    plot_embeddings,
    calculate_sampled_cosine_similarity,
    get_top_tokens,
    tokenize_data,
    CustomDataset,
    collate_fn,
    save_model,
    load_model
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

    # Load pre-trained model if the option is checked
    load_model_option = st.checkbox("Load Pre-trained Model")
    transformer_model, tokenizer = initialize_model(tokenizer_path="bpe_token.json", d_model=256)

    if load_model_option:
        model_path = st.text_input("Enter model path to load", "fine_tuned_transformer_model.pth")
        if model_path:
            load_model(transformer_model, model_path)
    
    # Tokenize the data
    st.write("Tokenizing data...")
    input_ids_batches, target_ids_batches = tokenize_data(tokenizer, data_dir)
    
    # Create DataLoader
    dataset = CustomDataset(tokenized_inputs=input_ids_batches, tokenized_targets=target_ids_batches)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)  # Assuming you have a val set
    
    # Fine-tune the model
    st.write("Fine-tuning the model...")
    loss_values, accuracy_values, perplexity_values, val_loss_values, val_accuracy_values = fine_tune_model(
        transformer_model, train_loader, val_loader, epochs=3, lr=5e-5
    )
        # Plot Training Loss
    st.write("Training Loss")
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(loss_values, label='Training Loss', color='blue')
    ax_loss.plot(val_loss_values, label='Validation Loss', color='orange')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    st.pyplot(fig_loss)

    # Plot Training Accuracy
    st.write("Training Accuracy")
    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(accuracy_values, label='Training Accuracy', color='blue')
    ax_acc.plot(val_accuracy_values, label='Validation Accuracy', color='orange')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    st.pyplot(fig_acc)

    # Plot Perplexity
    st.write("Training Perplexity")
    fig_perp, ax_perp = plt.subplots()
    ax_perp.plot(perplexity_values, label='Perplexity')
    ax_perp.set_xlabel('Epochs')
    ax_perp.set_ylabel('Perplexity')
    ax_perp.legend()
    st.pyplot(fig_perp)

    # Generate embeddings
    st.write("Generating embeddings...")
    embeddings = generate_embeddings(transformer_model, input_ids_batches)
    st.write(f"Total Embeddings: {embeddings.shape[0]}")  # Display total number of embeddings generated

    st.write("Saving the model...")
    save_model(transformer_model, "LuminaLM.pth")
    
    # Visualizations: Sample for PCA, t-SNE, and Cosine Similarity
    sample_size = st.slider("Select Sample Size for Visualization", min_value=1000, max_value=500000, value=1000)

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

    # Top Tokens
    st.write("Top Tokens")
    top_tokens = get_top_tokens(tokenizer, input_ids_batches)
    for token, freq in top_tokens:
        st.write(f"Token: {token}, Frequency: {freq}")
    
if __name__ == "__main__":
    main()
