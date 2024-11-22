Below is a **detailed documentation** file in Markdown format for the embeddings file we wrote. This file is suitable for a GitHub repository.

---

# **LuminaLM Embeddings Generator Documentation**

## **Overview**

The LuminaLM Embeddings Generator is designed to tokenize and process large datasets, generate embeddings using a custom transformer model, and store these embeddings in PineconeDB. Additionally, the system provides robust functionality for visualizing training metrics and embedding projections.

---

## **Key Features**

1. **Dataset Support**:
   - OpenWebText dataset.
   - Hugging Face medical datasets (e.g., PubMedQA, MedNLI, MIMIC Notes).
   - Local `.txt` files.

2. **Embeddings Generation**:
   - Tokenizes input text using a custom tokenizer.
   - Generates embeddings via a transformer model.
   - Stores embeddings in PineconeDB for scalable and efficient querying.

3. **Training and Evaluation**:
   - Implements fine-tuning with early stopping.
   - Tracks key training metrics such as loss, accuracy, and perplexity.

4. **Visualization**:
   - Training performance metrics are saved as plots.
   - Embedding distributions are visualized using PCA and t-SNE.

---

## **File Structure**

```plaintext
.
├── config.yaml             # Configuration file for model and training parameters
├── embeddings.py           # Main script for embeddings generation and visualization
├── pineconedb.py           # PineconeDB integration for storing embeddings
├── tokenizer.py            # Tokenizer training and loading utilities
├── dashboard.py            # Optional dashboard for real-time visualization
├── run_train.sh            # Bash script for running the training pipeline
├── requirements.txt        # Python dependencies
```

---

## **Setup and Installation**

### **Prerequisites**
- Python 3.8 or higher.
- Required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

### **Additional Dependencies**
- [PineconeDB API Key](https://www.pinecone.io/) for storing embeddings.

---

## **Usage**

### **1. Configuration**
Edit the `config.yaml` file to specify model and training parameters, such as:
- Batch size
- Learning rate
- Epochs
- Dataset split ratios
- Tokenizer and data paths

### **2. Generate Embeddings**
Run the `embeddings.py` script:
```bash
python embeddings.py --config config.yaml --tokenizer_path LuminaLM_text_tokens.json --local_data_dir path/to/local/data
```

### **3. Train and Fine-Tune**
To train the model and generate embeddings:
```bash
bash run_train.sh
```

### **4. Visualize Results**
Generated plots will be saved in the `plots/` directory:
- Loss, accuracy, and perplexity curves.
- Embedding projections using PCA and t-SNE.

---

## **Detailed Functionality**

### **Data Loading**

- **`load_openwebtext()`**:
  Loads the OpenWebText dataset and processes up to a defined sample size.
  
- **`load_medical_datasets()`**:
  Fetches data from Hugging Face medical datasets (e.g., PubMedQA, MedNLI).
  
- **`load_local_data(directory: str)`**:
  Reads and processes `.txt` files from a specified directory.

### **Tokenization**

- **Custom Dataset**:
  Handles tokenized input for model training.
  - Input and target sequences are padded using the `collate_fn` method.

- **Tokenizer**:
  The tokenizer is loaded from `LuminaLM_text_tokens.json`, or a new one can be trained using `tokenizer.py`.

### **Model Training**

- **Fine-Tuning**:
  Implements early stopping to avoid overfitting. Mixed precision training is supported for faster computation.

- **Training Components**:
  - **Optimizer**: AdamW
  - **Criterion**: CrossEntropyLoss

- **Metrics**:
  - Training and validation loss
  - Accuracy
  - Perplexity

### **Embeddings Generation**

- **`generate_embeddings()`**:
  - Produces embeddings for tokenized data.
  - Stores embeddings in PineconeDB under a specified index.

- **Integration with PineconeDB**:
  The `pineconedb.py` script provides functions for saving embeddings and querying them efficiently.

### **Visualization**

- **Metrics Visualization**:
  - `plot_training_loss`: Plots training loss over epochs.
  - `plot_training_accuracy`: Plots training accuracy over epochs.
  - `plot_training_perplexity`: Plots training perplexity over epochs.

- **Embedding Projections**:
  - `plot_embeddings_3d`: Generates 3D PCA and t-SNE projections of embeddings.

---

## **Configuration File**

The `config.yaml` file controls key aspects of training and data processing.

### **Sample Config**
```yaml
model:
  d_model: 512
  src_seq_len: 512
  batch_size: 128
  learning_rate: 5e-5
  epochs: 3
  patience: 3

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  max_samples: 100000

training:
  use_mixed_precision: true
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
```

---

## **Pipeline Workflow**

1. **Load and Process Data**:
   - Load datasets from OpenWebText, Hugging Face, and local sources.
   - Tokenize data using a custom tokenizer.

2. **Train Transformer Model**:
   - Fine-tune the model on tokenized data.
   - Monitor training metrics and save checkpoints.

3. **Generate Embeddings**:
   - Generate embeddings for input data.
   - Save embeddings to PineconeDB for downstream tasks.

4. **Visualize Results**:
   - Training metrics and embedding projections are saved as `.png` files.

---

## **Example Output**

### **Training Metrics**

1. **Loss Curve**:
   ![Training Loss](Plots/TrainingLoss.png)

2. **Accuracy Curve**:
   ![Training Accuracy](Plots/TrainingAccuracy.png)

3. **Perplexity Curve**:
   ![Training Perplexity](Plots/TrainingPerplexity.png)

### **Embedding Projections**

1. **PCA Visualization**:
   ![PCA Embeddings](Plots/PCA_Embeddings.png)

2. **t-SNE Visualization**:
   ![t-SNE Embeddings](Plots/TSNE_Embeddings.png)

---

## **Frequently Asked Questions**

### **1. How do I customize datasets?**
Edit the `config.yaml` file to update dataset paths or specify the `--local_data_dir` argument when running the script.

### **2. Where are embeddings stored?**
Embeddings are stored in PineconeDB under the index `luminalm-embeddings`. Ensure you have set up your Pinecone API key.

### **3. How do I resume training?**
Use the `--checkpoint` argument to specify a saved checkpoint:
```bash
python embeddings.py --config config.yaml --checkpoint path/to/checkpoint.pt
```

### **4. Can I change model parameters?**
Yes, modify the `config.yaml` file for model parameters like `d_model`, `src_seq_len`, and `batch_size`.

---

## **Contributing**

1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a Pull Request.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

```