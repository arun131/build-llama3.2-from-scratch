from llama.build_llama2 import Llama2
import torch
from types import SimpleNamespace

if __name__ == "__main__":
    # Define the configuration
    cfg = {
        "embed_dim": 512,
        "hidden_dim": 1024,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_len": 256,
        "dtype": torch.bfloat16,
        "vocab_size": 256,
    }

    cfg = SimpleNamespace(**cfg)

    assert cfg.embed_dim % cfg.num_heads == 0, "Embedding dimension must be divisible by the number of heads"
    assert (cfg.embed_dim // cfg.num_heads) % 2 == 0, "Head dimension must be even"
    # Initialize the model
    model = Llama2(cfg)
    
    # printing number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Define the input
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    print("Input", x)
    # Forward pass
    logits = model(x)

    print("Model output", logits.shape)