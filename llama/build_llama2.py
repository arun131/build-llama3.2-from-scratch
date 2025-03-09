import torch
import torch.nn as nn
from types import SimpleNamespace
from llama.building_blocks.transformer_block import RMSNorm, Llama2TransformerBlock

class Llama2(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        # check the configuration
        assert cfg.embed_dim % cfg.num_heads == 0, f"Embedding dimension must be divisible by the number of heads but the given value is {cfg.embed_dim} and the number of heads is {cfg.num_heads}"
        assert (cfg.embed_dim // cfg.num_heads) % 2 == 0, f"Embeding dimension must be even but the given value is {cfg.embed_dim}"

        self.embed_dim = cfg.embed_dim
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.dtype = cfg.dtype
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)
        self.transformer_blocks = nn.Sequential(
                                        *[Llama2TransformerBlock(cfg) for _ in range(cfg.num_layers)]
                                        )
        self.final_norm = RMSNorm(cfg.embed_dim, dtype=cfg.dtype)
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, dtype=cfg.dtype, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement KV caching for faster inference
        """Forward pass of the Llama2 model
        It takes the input of batch, seq_len of input token ids and returns the logits

        Args:
            x (torch.Tensor): input token ids of shape (batch, seq_len)

        Returns:
            torch.Tensor: logits of shape (batch, seq_len, vocab_size)
        """
        x = self.token_embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x
