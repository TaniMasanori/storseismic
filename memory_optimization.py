import torch
import gc
import os

def setup_memory_optimization():
    """Set up memory optimization configurations"""
    # PyTorch CUDA memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable memory efficient attention if available
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

def clear_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        return allocated, reserved
    return 0, 0

def monitor_memory(stage=""):
    """Monitor and print memory usage"""
    allocated, reserved = get_memory_usage()
    print(f"{stage} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB") 