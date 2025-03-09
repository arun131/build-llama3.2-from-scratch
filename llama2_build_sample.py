from llama.build_llama2 import Llama2
import torch
from types import SimpleNamespace
from utils.model_utils import get_device
from utils.generation_utils import text_to_token_ids, token_ids_to_text, generate
from huggingface_hub import login, hf_hub_download
from dotenv import load_dotenv
import os
load_dotenv()

if __name__ == "__main__":

    # Get the device
    device = get_device()

    # Define the configuration
    # Actual Llama2 configuration is:
    # LLAMA2_CONFIG_7B = {
    #     "vocab_size": 32000,       
    #     "max_seq_len": 4096,       
    #     "embed_dim": 4096,         
    #     "num_heads": 32,           
    #     "num_layers": 32,          
    #     "hidden_dim": 11008,       
    #     "dtype": torch.bfloat16    
    # }
    cfg = SimpleNamespace(**{
        "embed_dim": 512,               # Embedding dimension
        "hidden_dim": 1024,             # Size of the intermediate dimension in FeedForward
        "num_heads": 4,                 # Number of attention heads
        "num_layers": 2,                # Number of layers of the transformer blocks
        "max_seq_len": 256,             # Context length the number of tokens in the input sequence
        "dtype": torch.bfloat16,        # Lower-precision dtype to reduce memory usage    
        "vocab_size": 256,              # Vocabulary size
    })
 
    # Initialize the model
    model = Llama2(cfg).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Define random input of 2 X 10 size (batch size X sequence length)
    batch_size = 2
    sequence_length = 10
    x = torch.randint(0, cfg.vocab_size, (batch_size, sequence_length)).to(device)

    # Download the tokenizer from Hugging Face Hub
    login(access_token=os.getenv("HF_ACCESS_TOKEN"))
    tokenizer_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b",
        filename="tokenizer.model",
        local_dir="Llama-2-7b"
        )

    


    print("Input", x)
    # Forward pass
    logits = model(x)

    print("Model output", logits.shape)