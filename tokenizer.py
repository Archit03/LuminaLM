from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
import os

def create_tokens(vocab_size=20000, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<eot>", "<bos>", "<eos>"]
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()  # Better subword and punctuation handling
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    return tokenizer, trainer

def preprocess_csv(file):
    # Custom preprocessing for CSV files
    import pandas as pd
    df = pd.read_csv(file)
    text_data = " ".join(df.astype(str).values.flatten())
    return text_data

def preprocess_files(files):
    all_text = ""
    for file in files:
        if file.endswith(".csv"):
            all_text += preprocess_csv(file)
        elif file.endswith(".txt"):
            with open(file, 'r') as f:
                all_text += f.read()
    return all_text

def train_tokenizer(tokenizer, trainer, files):
    text_data = preprocess_files(files)
    if text_data.strip() == "":
        raise ValueError("No valid text data found for training.")
    tokenizer.train_from_iterator([text_data], trainer)

def save_tokenizer(tokenizer, path="bpe_token.json"):
    tokenizer.save(path)

def load_tokenizer(path="bpe_token.json"):
    return Tokenizer.from_file(path)

# Example Usage
files = ["data.txt", "healthcare_dataset.csv", "ACS_CA3_Book.txt", "Genomes_3 - T.A. Brown_.txt", "train.txt", "test.txt"]
tokenizer, trainer = create_tokens(vocab_size=199997)
train_tokenizer(tokenizer, trainer, files)
save_tokenizer(tokenizer)
