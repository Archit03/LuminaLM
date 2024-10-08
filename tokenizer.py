from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

def create_tokens(vocab_size=20000, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<eot>", "<bos>", "<eos>"]
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    return tokenizer, trainer

def train_tokenizer(tokenizer, trainer, files):
    valid_files = [file for file in files if os.path.isfile(file)]
    if not valid_files:
        raise ValueError("No valid files found for training.")
    tokenizer.train(valid_files, trainer)

def save_tokenizer(tokenizer, path="bpe_token.json"):
    tokenizer.save(path)

def load_tokenizer(path="bpe_token.json"):
    return Tokenizer.from_file(path)

# Example Usage
files = ["data.txt", "healthcare_dataset.csv", "ACS_CA3_Book.txt", "Genomes_3 - T.A. Brown_.txt", "train.txt", "test.txt"]
tokenizer, trainer = create_tokens(vocab_size=199997)
train_tokenizer(tokenizer, trainer, files)
save_tokenizer(tokenizer)