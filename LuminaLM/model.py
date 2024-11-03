import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from tqdm import tqdm
import embeddings.embeddings as embeddings  # Custom embeddings logic in embeddings.py
import embeddings.tokenizer as tokenizer  # Custom tokenizer from tokenizer.py

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
        self.act = nn.GELU()
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
class LuminaLMConfig:
    block_size: int = 1024
    vocab_size: int = 50_000
    n_layer: int = 120
    n_head: int = 64
    n_embd: int = 512  

# LuminaLM Model Class
class LuminaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load custom embeddings from embeddings.py
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': embeddings.load_pretrained_embeddings(),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tokenizer = tokenizer.load_tokenizer()

    def forward(self, input_ids):
        device = input_ids.device
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for block in self.transformer['h']:
            hidden_states = block(hidden_states)

        hidden_states = self.transformer['ln_f'](hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate_text(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt)
        device = next(self.parameters()).device
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(input_ids)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=1)

        output_text = self.tokenizer.decode(input_ids[0].tolist())
        return output_text

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.tokenizer.encode(self.texts[idx])
        target_ids = input_ids[1:]
        return torch.tensor(input_ids[:-1], dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

if __name__ == "__main__":
    config = LuminaLMConfig()
    model = LuminaLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    texts = ["Sample sentence for dataset.", "Another sample for the model."]
    dataset = TextDataset(texts, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    epochs = 3

    for epoch in range(epochs):
        model.train()
        loop = tqdm(dataloader, leave=True)
        for input_ids, target_ids in loop:
            optimizer.zero_grad()
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            outputs = model(input_ids)
            logits = outputs[..., :-1, :].contiguous()
            shift_labels = target_ids[..., :-1].contiguous()
            loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    model.eval()
    prompt = input("Please input the prompt.")
    generated_text = model.generate_text(prompt, max_length=50)
    print("Generated Text:", generated_text)
