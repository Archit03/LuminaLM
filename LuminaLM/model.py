import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Optional, Tuple, Dict, Union, List
import logging
from dataclasses import dataclass
import json
import math
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LuminaLMConfig:
    """Configuration for Encoder-Decoder LuminaLM model."""
    n_embd: int = 512
    n_head: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
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
    shared_embeddings: bool = True
    tie_word_embeddings: bool = True
    use_cache: bool = True
    use_rotary_embeddings: bool = True
    fp16: bool = False
    max_grad_norm: float = 1.0
    advanced_attention: bool = False  # Support for advanced attention mechanisms

    @classmethod
    def from_json(cls, json_file: str) -> 'LuminaLMConfig':
        """Load configuration from a JSON file."""
        try:
            with open(json_file, 'r') as f:
                return cls(**json.load(f))
        except FileNotFoundError:
            logger.error(f"Configuration file '{json_file}' not found.")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file '{json_file}'.")
            raise

    def to_json(self, json_file: str) -> None:
        """Save configuration to a JSON file."""
        try:
            with open(json_file, 'w') as f:
                json.dump(self.__dict__, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save configuration to '{json_file}': {e}")
            raise

    def validate(self) -> None:
        """Validate configuration parameters."""
        logger.debug("Validating configuration parameters.")
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.block_size <= self.max_position_embeddings, "block_size cannot exceed max_position_embeddings"
        assert self.block_size > 0, "block_size must be positive"
        assert self.n_head > 0, "n_head must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert 0.0 <= self.embd_pdrop <= 1.0, "embd_pdrop must be between 0 and 1"
        assert 0.0 <= self.resid_pdrop <= 1.0, "resid_pdrop must be between 0 and 1"
        assert 0.0 <= self.attn_pdrop <= 1.0, "attn_pdrop must be between 0 and 1"
        logger.info("Configuration parameters are valid.")

# Rotary Embeddings Helper Functions
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the dimensions of the tensor for rotary embeddings."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to the query and key tensors.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        cos (torch.Tensor): Cosine component of the rotary embeddings.
        sin (torch.Tensor): Sine component of the rotary embeddings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing updated query and key tensors.
    """
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryEmbedding(nn.Module):
    """Rotary Embeddings for enhanced positional encoding."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached rotary embeddings for a given sequence length.

        Args:
            seq_len (int): Length of the sequence.
            device (torch.device): Device to place embeddings on.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine components of rotary embeddings.
        """
        if seq_len > self.max_seq_len_cached:
            raise ValueError(f"Requested sequence length {seq_len} exceeds max cached length {self.max_seq_len_cached}.")
        return self.cos_cached[:, :, :seq_len, :].to(device), self.sin_cached[:, :, :seq_len, :].to(device)

# Position Embeddings Layer
class PositionEmbeddings(nn.Module):
    """Learned positional embeddings for the model."""
    def __init__(self, config: LuminaLMConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for position embeddings.

        Args:
            position_ids (torch.Tensor): Tensor of position IDs.

        Returns:
            torch.Tensor: Positional embeddings.
        """
        return self.dropout(self.embeddings(position_ids))

# Flash Attention Layer with Rotary Embedding Support
class FlashAttention(nn.Module):
    """FlashAttention with support for rotary embeddings for efficient memory usage."""
    def __init__(self, config: LuminaLMConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.attn_pdrop)
        
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()

        # Input validation for tensor types and shapes
        if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("Query, key, and value must all be torch.Tensor.")
        if query.shape != key.shape or key.shape != value.shape:
            raise ValueError("Query, key, and value must have the same shape.")
        if mask is not None and mask.shape != (batch_size, 1, 1, seq_len):
            raise ValueError(f"Invalid attention mask shape. Expected ({batch_size}, 1, 1, {seq_len}), got {mask.shape}")

        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if enabled
        if hasattr(self, 'rotary_emb') and position_ids is not None:
            cos, sin = self.rotary_emb(seq_len, query.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        with autocast(enabled=True):
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.head_dim)
        output = self.out_proj(context)

        return output

# Feed Forward Layer of Transformer
class FeedForward(nn.Module):
    """Feed Forward Layer of the Transformer."""
    def __init__(self, config: LuminaLMConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feed-forward computation.
        """
        with autocast(enabled=True):
            x = self.gelu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
        return x

# Encoder Block
class EncoderBlock(nn.Module):
    """Encoder block consisting of multi-head attention and feed-forward layers."""
    def __init__(self, config: LuminaLMConfig):
        super().__init__()
        self.attn = FlashAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.use_checkpoint = config.use_checkpoint

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for encoder block.

        Args:
            x (torch.Tensor): Input tensor.
            mask (Optional[torch.Tensor]): Attention mask.
            position_ids (Optional[torch.Tensor]): Positional IDs for rotary embeddings.

        Returns:
            torch.Tensor: Encoder output.
        """
        def _forward(x: torch.Tensor) -> torch.Tensor:
            with autocast(enabled=True):
                attn_output = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask, position_ids)
                x = x + attn_output
                x = x + self.ff(self.ln2(x))
            return x

        if self.use_checkpoint and x.requires_grad:
            return checkpoint(_forward, x)
        return _forward(x)

# Decoder Block
class DecoderBlock(nn.Module):
    """Decoder block consisting of self-attention, cross-attention, and feed-forward layers."""
    def __init__(self, config: LuminaLMConfig):
        super().__init__()
        self.self_attn = FlashAttention(config)
        self.cross_attn = FlashAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln3 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.use_checkpoint = config.use_checkpoint

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for decoder block.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Encoder output for cross-attention.
            self_mask (Optional[torch.Tensor]): Self-attention mask.
            cross_mask (Optional[torch.Tensor]): Cross-attention mask.
            position_ids (Optional[torch.Tensor]): Positional IDs for rotary embeddings.

        Returns:
            torch.Tensor: Decoder output.
        """
        def _forward(x: torch.Tensor) -> torch.Tensor:
            with autocast(enabled=True):
                # Self-attention layer
                self_attn_output = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=self_mask, position_ids=position_ids)
                x = x + self_attn_output

                # Cross-attention layer
                cross_attn_output = self.cross_attn(self.ln2(x), encoder_output, encoder_output, mask=cross_mask, position_ids=position_ids)
                x = x + cross_attn_output

                # Feed-forward layer
                x = x + self.ff(self.ln3(x))
            return x

        if self.use_checkpoint and x.requires_grad:
            return checkpoint(_forward, x)
        return _forward(x)

# LuminaLM Model
class LuminaLM(nn.Module):
    """Encoder-Decoder Transformer model: LuminaLM."""
    def __init__(self, config: LuminaLMConfig):
        super().__init__()
        self.config = config
        config.validate()

        # Embedding layers
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = PositionEmbeddings(config)
        self.drop = nn.Dropout(config.embd_pdrop)

        # Encoder
        self.encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder_layers)])
        self.encoder_ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Decoder
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_decoder_layers)])
        self.decoder_ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Output head
        if config.shared_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Custom weight initialization with variance scaling based on layer depth."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            layer_depth = self.get_depth(module)
            std_dev = self.config.initializer_range / math.sqrt(layer_depth + 1)
            module.weight.data.normal_(mean=0.0, std=std_dev)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_depth(self, module: nn.Module) -> int:
        """Get the depth of a module within the network."""
        if isinstance(module, (EncoderBlock, DecoderBlock)):
            return 1
        return 0

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the LuminaLM model."""
        device = input_ids.device

        # Input Validation
        if not isinstance(input_ids, torch.Tensor) or not isinstance(decoder_input_ids, torch.Tensor):
            raise TypeError("Input tensors must be of type torch.Tensor.")
        if input_ids.dim() != 2 or decoder_input_ids.dim() != 2:
            raise ValueError("Input tensors must be of rank 2 (batch_size, seq_len).")

        # Encode if encoder_outputs not provided
        encoder_embeddings = self.drop(self.wte(input_ids) + self.position_embeddings(input_ids))
        encoder_hidden_states = encoder_embeddings

        for layer in self.encoder:
            encoder_hidden_states = layer(encoder_hidden_states, attention_mask)

        encoder_outputs = self.encoder_ln(encoder_hidden_states)

        # Decode
        decoder_embeddings = self.drop(self.wte(decoder_input_ids) + self.position_embeddings(decoder_input_ids))
        decoder_hidden_states = decoder_embeddings

        for idx, decoder_layer in enumerate(self.decoder):
            decoder_hidden_states = decoder_layer(decoder_hidden_states, encoder_outputs, self_mask=decoder_attention_mask)

        decoder_outputs = self.decoder_ln(decoder_hidden_states)
        logits = self.lm_head(decoder_outputs)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        early_stopping: bool = True,
    ) -> torch.Tensor:
        """Generate text from the model given an input prompt."""
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("Input tensor must be of type torch.Tensor.")
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer.")

        batch_size = input_ids.size(0)
        generated_tokens = input_ids

        for _ in range(max_length):
            logits = self.forward(input_ids=generated_tokens, decoder_input_ids=generated_tokens)

            # Apply temperature
            logits = logits[:, -1, :] / temperature

            # Top-K and top-p filtering
            if top_k is not None:
                logits = self.top_k_filtering(logits, top_k)
            if top_p is not None:
                logits = self.top_p_filtering(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            if early_stopping and (next_token == self.config.eos_token_id).all():
                break

        return generated_tokens

    @staticmethod
    def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits using top-K filtering."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    @staticmethod
    def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits using top-p (nucleus) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits

