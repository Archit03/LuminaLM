import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import load_dataset
from tqdm import tqdm
import embeddings.embeddings as embeddings  # Custom embeddings logic
import embeddings.tokenizer as tokenizer  # Custom tokenizer logic

# Causal Self-Attention Layer
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
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

# Multi-Layer Perceptron
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

# LuminaLM Configuration
class LuminaLMConfig:
    block_size: int = 1024
    vocab_size: int = 50_000
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 768

# LuminaLM Model
class LuminaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(input_ids)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=1)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output_text

# Hugging Face Dataset Preparation
def prepare_dataset(dataset_name, split, tokenizer, block_size=1024):
    dataset = load_dataset(dataset_name, split=split)
    
    def tokenize_function(examples):
        return tokenizer.batch_encode_plus(examples['text'], truncation=True, max_length=block_size)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids'])
    return tokenized_dataset

if __name__ == "__main__":
    config = LuminaLMConfig()
    model = LuminaLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tokenizer = tokenizer.load_tokenizer()

    # Load and prepare dataset
    block_size = config.block_size
    dataset = prepare_dataset("wikitext", "train", tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    epochs = 3

    for epoch in range(epochs):
        model.train()
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    model.eval()
    prompt = input("Provide a task description: ")
    generated_text = model.generate_text(prompt, max_length=50)
    print("Generated Text:", generated_text)
