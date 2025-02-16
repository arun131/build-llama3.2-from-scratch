import torch
import torch.nn as nn
from types import SimpleNamespace

class RMSNorm(nn.Module):
    # RMSNorm from to replace the LayerNorm in GPT2
    # https://arxiv.org/abs/1910.07467
    def __init__(self, embed_dim: int, dtype: torch.dtype = torch.float, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.rms_weights = nn.Parameter(torch.ones(embed_dim), dtype= dtype)
    
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
    # The description of the layer can be found in the paper
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

class LlamaTransformerBlock(nn.Module):
    @classmethod
    def compute_rotation_angles(cls, head_dim: int, dtype: torch.dtype, max_seq_len: int = 4096, base_theta: int = 10000):
        """Precompute the rotary position embeddings
        Adapted from https://github.com/rasbt/LLMs-from-scratch/blob/29353c74d8e9a8a55064526fe53456c487b707eb/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb
        """
        freq_seq = torch.arange(0, head_dim, 2.0)[: head_dim // 2]
        inv_freq = 1 / (base_theta ** (freq_seq / head_dim))
        positions = positions = torch.arange(max_seq_len)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        angles = torch.cat([angles, angles], dim=1)
        # Precompute sine and cosine
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin
    
    @classmethod
    def 
