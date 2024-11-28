import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from transformers import AdamW
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Import your custom tokenizer
from embeddings.tokenizer import load_tokenizer
from embeddings.pineconedb import fetch_embeddings  # Using the fetch_embeddings function as requested

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

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

class LuminaLMConfig:
    def __init__(self):
        self.block_size = 1024
        self.vocab_size = 60000  # Updated to match your tokenizer's vocab size
        self.n_layer = 28
        self.n_head = 16
        self.n_embd = 1152
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.epochs = 5  # Added epochs to config

class LuminaLM(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.transformer = nn.ModuleDict({
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.embd_pdrop),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Fetch embeddings using the fetch_embeddings function
        inputs_embeds = fetch_embeddings(input_ids)
        pos_embeds = self.transformer['wpe'](position_ids)
        hidden_states = inputs_embeds + pos_embeds
        hidden_states = self.transformer['drop'](hidden_states)

        for block in self.transformer['h']:
            hidden_states = block(hidden_states)

        hidden_states = self.transformer['ln_f'](hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate_text(self, prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
        self.eval()
        device = next(self.parameters()).device
        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(input_ids)
                logits = outputs[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits.scatter_(dim=-1, index=indices_to_remove, value=float('-inf'))
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.token_to_id('</s>'):  # Using '</s>' as EOS token
                    break

        output_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return generated_text

def train_model(model, train_loader, val_loader, config, device):
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.epochs)
    scaler = GradScaler()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.tokenizer.token_to_id('<pad>'))
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                outputs = model(input_ids)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                val_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def collate_fn(batch, tokenizer):
    input_ids = [tokenizer.encode(example['text']).ids for example in batch]
    input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.token_to_id('<pad>'))
    return {'input_ids': input_ids}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = LuminaLMConfig()
    tokenizer = load_tokenizer('medical_tokenizer.json')  # Specify the path to your tokenizer file
    model = LuminaLM(config, tokenizer).to(device)
    
    # Dataset preparation
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    train_model(model, train_loader, val_loader, config, device)
    
    # Model testing/generation
    prompt = input("Enter a prompt for text generation: ")
    generated_text = model.generate_text(prompt, max_length=100)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
