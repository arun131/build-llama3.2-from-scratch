import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def calculate_memory_requirements(model: torch.nn.Module, dtype: torch.dtype) -> int:
    """
    Calculate the memory requirements of a model given the dtype of the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate buffer size 
    total_buffer_size = sum(buffer.numel() for buffer in model.buffers())

    # Calculate the memory requirements
    memory_requirements = (total_params + total_grads + total_buffer_size) * dtype.itemsize / ()

    print(f"Total number of parameters: {total_params:,}")
    print(f"Memory requirements: {memory_requirements/(1024**3):.2f} Giga bytes")

    return memory_requirements


