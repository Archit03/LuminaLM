import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
import embeddings  # Custom embeddings logic in embeddings.py
from tqdm import tqdm
import tokenizer  # Custom tokenizer from tokenizer.py

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

# Configuration for the LuminaLM Model
@dataclass
class LuminaLMConfig:
    block_size: int = 1024  # Number of tokens in a sequence
    vocab_size: int = 199997  # Number of tokens in the vocabulary
    n_layer: int = 48  # Number of layers
    n_head: int = 48  # Number of attention heads
    n_embd: int = 6144  # Embedding dimension

# LuminaLM Model Class
class LuminaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load custom embeddings from embeddings.py
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': embeddings.load_pretrained_embeddings(),  # Load custom embeddings
            'wpe': nn.Embedding(config.block_size, config.n_embd),  # Positional encodings
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tokenizer = tokenizer.load_tokenizer()  # Load custom tokenizer

    def forward(self, input_ids):
        device = input_ids.device
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Get word embeddings from embeddings.py
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
        # Tokenize the prompt using the custom tokenizer from tokenizer.py
        input_ids = self.tokenizer.encode(prompt)

        # Convert to tensor and move to the correct device
        device = next(self.parameters()).device
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

        # Pass the tokenized input into the model and generate output tokens
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(input_ids)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=1)

        # Decode using your custom tokenizer
        output_text = self.tokenizer.decode(input_ids[0].tolist())
        return output_text

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.tokenizer.encode(self.texts[idx])
        target_ids = input_ids[1:]  # Next token prediction task
        return torch.tensor(input_ids[:-1], dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

# Example usage with LuminaLM model
if __name__ == "__main__":
    model_config = LuminaLMConfig()  # Initialize custom GPT configuration
    model = LuminaLM(model_config)  # Initialize LuminaLM model

    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)

    # Prepare your dataset
    texts = ["The medical diagnosis reveals a pattern.", "The MRI scan shows abnormal growth."]
    dataset = TextDataset(texts, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Training Loop
    model.train()  # Set the model to training mode
    epochs = 3  # Number of epochs

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for input_ids, target_ids in loop:
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            outputs = model(input_ids)

            # Shift logits for next-token prediction
            logits = outputs[..., :-1, :].contiguous()  # Keep logits unchanged
            shift_labels = target_ids[..., :-1].contiguous()  # Shift labels

            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1))

            # Backpropagation
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    # Generate text after training
    model.eval()  # Set model to eval mode for generation
    prompt = "The patient exhibits"
    generated_text = model.generate_text(prompt, max_length=50)
    print("Generated Text:", generated_text)
