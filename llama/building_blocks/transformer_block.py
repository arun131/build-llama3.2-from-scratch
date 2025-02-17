import torch
import torch.nn as nn
from types import SimpleNamespace
from llama.building_blocks.rotary_position_embedding import RotaryPositionEmbeddings

class RMSNorm(nn.Module):
    # RMSNorm from to replace the LayerNorm in GPT2
    # https://arxiv.org/abs/1910.07467
    def __init__(self, embed_dim: int, dtype: torch.dtype = torch.float, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.rms_weights = nn.Parameter(torch.ones(embed_dim)).to(dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(means + self.eps)
        x = x * self.rms_weights
        return x.to(dtype=x.dtype)

class Silu(nn.Module):
    # Signmoid Linear Unit (SiLU)
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class FeedForwardNN(nn.Module):
    # feed forward layer for the Llama transformer block
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.linear1 = nn.Linear(cfg.embed_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)
        self.activation = Silu()
        self.linear2 = nn.Linear(cfg.embed_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)
        self.linear3 = nn.Linear(cfg.hidden_dim, cfg.embed_dim, dtype=cfg.dtype, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1 = self.linear1(x)
        fc2 = self.linear2(x)
        x = self.activation(fc1) * fc2
        return self.linear3(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.W_query = nn.Linear(self.embed_dim , self.embed_dim , bias=False, dtype=cfg.dtype)
        self.W_key = nn.Linear(self.embed_dim , self.embed_dim , bias=False, dtype=cfg.dtype)
        self.W_value = nn.Linear(self.embed_dim , self.embed_dim , bias=False, dtype=cfg.dtype)
        self.proj_out = nn.Linear(self.embed_dim , self.embed_dim , bias=False, dtype=cfg.dtype)

        self.register_buffer("mask", torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1))

        cos, sin = RotaryPositionEmbeddings.compute_rotation_angles(self.head_dim, cfg.max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        key = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        # change the dimension of the input to batch, num_heads, seq_len, head_dim
        keys = key.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotation to keys and queries
        keys = RotaryPositionEmbeddings.compute_rope(keys, self.cos, self.sin)
        queries = RotaryPositionEmbeddings.compute_rope(queries, self.cos, self.sin)

        # Compute the attention scores
        attention_scores = queries @ keys.transpose(-2, -1)
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(mask_bool, float("-inf"))
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vector = (attention_weights @ values).transpose(1, 2)
        
        # Reshape the context vector to the original shape
        context_vector = context_vector.reshape(batch, seq_len, self.embed_dim)
        return self.proj_out(context_vector) # final projection


class Llama2TransformerBlock(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.attention = MultiHeadedAttention(cfg)
        self.norm1 = RMSNorm(cfg.embed_dim, dtype=cfg.dtype)
        self.norm2 = RMSNorm(cfg.embed_dim, dtype=cfg.dtype)
        self.ffn = FeedForwardNN(cfg)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        # first part through the attention layer and add skip connection
        x = self.norm1(x)
        x = self.attention(x)
        x = x + x_skip
        x_skip = x

        # second part through the feed forward layer and add skip connection
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + x_skip
        return x
