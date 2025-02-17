from llama.build_llama2 import Llama2
import torch
from types import SimpleNamespace
from utils.model_utils import get_device
from utils.generation_utils import text_to_token_ids, token_ids_to_text, generate

if __name__ == "__main__":

    # Get the device
    device = get_device()

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
    model = Llama2(cfg).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Define the input
    x = torch.randint(0, cfg.vocab_size, (2, 10)).to(device)
    print("Input", x)
    # Forward pass
    logits = model(x)

    print("Model output", logits.shape)