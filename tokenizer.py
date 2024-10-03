from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a tokenizer with BPE model
tokenizer = Tokenizer(BPE())

# Configure the tokenizer to split by whitespace
tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer on your dataset
trainer = BpeTrainer(vocab_size=199997, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
files = ["data.txt", "healthcare_dataset.csv"]  # Path to your dataset
tokenizer.train(files, trainer)

# Save the tokenizer
tokenizer.save("bpe_token.json")
