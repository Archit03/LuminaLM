import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for LuminaLM model parameters."""
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 12
    block_size: int = 128
    vocab_size: int = 60000
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    use_checkpoint: bool = True
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_position_embeddings: int = 128

    @classmethod
    def from_json(cls, json_file: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str):
        """Save configuration to JSON file."""
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(self.__dict__, f, indent=2)

class LuminaLM(nn.Module):
    """Main LuminaLM model class."""
    def __init__(self, config: ModelConfig, tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.max_position_embeddings, config.n_embd),
            'drop': nn.Dropout(config.embd_pdrop),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
        })
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def from_pretrained(cls, pretrained_dir: str) -> 'LuminaLM':
        """Load pretrained model from directory."""
        config = ModelConfig.from_json(os.path.join(pretrained_dir, 'config.json'))
        tokenizer = Tokenizer.from_file(os.path.join(pretrained_dir, 'tokenizer.json'))
        model = cls(config, tokenizer)
        model.load_state_dict(torch.load(os.path.join(pretrained_dir, 'pytorch_model.bin')))
        return model

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None) -> torch.Tensor:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Get embeddings
        token_embeddings = self.transformer.wte(input_ids)
        position_embeddings = self.transformer.wpe(position_ids)
        hidden_states = self.transformer.drop(token_embeddings + position_embeddings)

        # Process through transformer blocks
        for i, block in enumerate(self.transformer.h):
            hidden_states = block(hidden_states)

        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    @torch.no_grad()
    def generate_text(self, prompt: str, max_length: int = 50, 
                      temperature: float = 0.7, top_k: int = 10) -> List[str]:
        """
        Generate text using various sampling strategies.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Tokenize prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt).ids, device=device).unsqueeze(0)
        
        generated_tokens = []
        
        for _ in range(max_length):
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            if next_token.item() == self.config.eos_token_id:
                break
        
        return [self.tokenizer.decode(generated_tokens, skip_special_tokens=True)]

# Function to load the model and generate text based on a prompt
def generate_response(model_dir: str, prompt: str, pretrained_embeddings: Optional[str] = None,
                      max_length: int = 50, top_k: int = 10, temperature: float = 0.7):
    # Load configuration
    config_path = f"{model_dir}/config.json"
    config = ModelConfig.from_json(config_path)

    # Load tokenizer
    tokenizer_path = f"{model_dir}/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load model
    model = LuminaLM.from_pretrained(model_dir)
    
    # Load pre-trained embeddings if specified
    if pretrained_embeddings:
        model.load_trained_embeddings(pretrained_embeddings)
    
    model.eval()

    # Check for CUDA availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate text
    generated_texts = model.generate_text(prompt, max_length=max_length, temperature=temperature, top_k=top_k)
    return generated_texts[0]  # Assuming num_return_sequences=1 for simplicity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a response using a pre-trained LuminaLM model")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing the trained model")
    parser.add_argument("--prompt", type=str, required=True, help="The input prompt for generating the response")
    parser.add_argument("--pretrained_embeddings", type=str, help="Path to pre-trained embeddings file", default=None)
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the generated response")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling for text generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    args = parser.parse_args()

    # Generate response
    response = generate_response(
        model_dir=args.model_dir,
        prompt=args.prompt,
        pretrained_embeddings=args.pretrained_embeddings,
        max_length=args.max_length,
        top_k=args.top_k,
        temperature=args.temperature
    )

    # Print the generated response
    print("Prompt: ", args.prompt)
    print("Generated Response: ", response)