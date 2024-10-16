from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import os

# Create a function to handle tokenization
def create_tokens(vocab_size=20000, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<eot>", "<bos>", "<eos>"]
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()  # Better subword and punctuation handling
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    return tokenizer, trainer

# Custom function for preprocessing CSV files
def preprocess_csv(file):
    import pandas as pd
    df = pd.read_csv(file)
    text_data = " ".join(df.astype(str).values.flatten())
    return text_data

# Function to preprocess files from a directory
def preprocess_files_from_directory(directory):
    all_text = ""
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".csv"):
                all_text += preprocess_csv(file_path)
            elif file_path.endswith(".txt"):
                # Open the file with utf-8 encoding
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    all_text += f.read()
    return all_text

# Train the tokenizer on preprocessed text
def train_tokenizer(tokenizer, trainer, directory):
    text_data = preprocess_files_from_directory(directory)
    if text_data.strip() == "":
        raise ValueError("No valid text data found for training.")
    tokenizer.train_from_iterator([text_data], trainer)

# Save the tokenizer to a file
def save_tokenizer(tokenizer, path="bpe_token.json"):
    tokenizer.save(path)

# Load the tokenizer from a saved file
def load_tokenizer(path="bpe_token.json"):
    return Tokenizer.from_file(path)

# Example Usage
if __name__ == "__main__":
    directory = r'/home/ubuntu/LuminaLM/Data'  # Adjust to your actual directory path
    tokenizer, trainer = create_tokens(vocab_size=199997)
    train_tokenizer(tokenizer, trainer, directory)
    save_tokenizer(tokenizer)
