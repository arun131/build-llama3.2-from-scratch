import torch
import torch.nn as nn

class RotaryPositionEmbeddings(nn.Module):
    @classmethod
    def compute_rotation_angles(cls, head_dim: int, max_seq_len: int = 4096, base_theta: int = 10000):
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
    def apply_rotation(cls, x: torch.Tensor) -> torch.Tensor:
        """Split x into first half and second half"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim = -1)
    
    @classmethod
    def compute_rope(cls, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Compute the rotary position embedding"""
        _, _, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"

        rotated = cls.apply_rotation(x)

        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        # Apply the rotary transformation
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.to(dtype=x.dtype)
