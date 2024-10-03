from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2Tokenizer
from tqdm import tqdm
import tokenizer

# Custom Causal Self-Attention Layer
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # For q, k, v
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

# Multi-Layer Perceptron (Feed-Forward Network)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()  # Activation function
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Transformer Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Configuration for the Sentient Sculptor Model
@dataclass
class SentientSculptorConfig:
    block_size: int = 1024  # Number of tokens in a sequence
    vocab_size: int = 199997  # Number of tokens in the vocabulary
    n_layer: int = 48  # 12 layers (adjust as needed)
    n_head: int = 48 # Number of attention heads
    n_embd: int = 6144  # Embedding dimension

# Sentient Sculptor Model Class
class SentientSculptor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def forward(self, input_ids):
        device = input_ids.device
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Get word embeddings
        inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Forward pass through transformer blocks
        for block in self.transformer['h']:
            hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.transformer['ln_f'](hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)
        return logits

    def generate_text(self, prompt, max_length=50):
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # Move the input to the correct device if necessary
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        # Pass the tokenized input into the model and generate output tokens
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(input_ids)
                logits = outputs[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        # Decode the generated tokens into text
        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output_text

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage with Sentient Sculptor model
model_config = SentientSculptorConfig()  # Initialize a smaller custom GPT configuration
model = SentientSculptor(model_config)  # Randomly initialized Sentient Sculptor model

total_params = count_parameters(model)
print(f'Total number of trainable parameters: {total_params}')

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


# Get user prompt
prompt = input("Enter a prompt: ")
generated_text = model.generate_text(prompt, max_length=50)
print(generated_text)