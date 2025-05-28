# Safe Data Loading Cell - Replacement for the problematic cell in nb1_original_DAS.ipynb
# This cell replaces the original data loading code that caused kernel crashes

import torch
import os
import gc
import psutil
from torch.serialization import add_safe_globals
from storseismic.utils import SSDataset
from memory_safe_data_loader import create_safe_data_loaders

# Clear any existing data and free memory
torch.cuda.empty_cache()
gc.collect()

print("=== Memory-Safe Data Loading ===")
print(f"Initial memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024:.2f}GB")

# Add SSDataset to the safe globals list (as in original code)
add_safe_globals([SSDataset])

# Configuration from the notebook
config_dataset = './data/pretrain/'
config_batch_size = 256

# Check available system memory
available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024

print(f"System memory: {available_memory:.2f}GB available / {total_memory:.2f}GB total")

# Determine safe memory limit (use 70% of available memory)
safe_memory_limit = min(available_memory * 0.7, 16.0)  # Cap at 16GB
print(f"Setting memory limit to: {safe_memory_limit:.2f}GB")

try:
    # Create memory-safe data loaders
    train_dataloader, test_dataloader, data_loader = create_safe_data_loaders(
        data_path=config_dataset,
        batch_size=config_batch_size,
        max_memory_gb=safe_memory_limit
    )
    
    # Set visualization parameters (from original code)
    vmin_all = -1
    vmax_all = 1
    
    print("\n=== Data Loading Successful ===")
    print(f"Training data loader: {len(train_dataloader)} batches")
    print(f"Test data loader: {len(test_dataloader)} batches")
    print(f"Batch size: {config_batch_size}")
    
    # Get a sample batch to verify data structure
    sample_batch = data_loader.get_sample_batch(train_dataloader)
    if sample_batch:
        print(f"Sample batch keys: {list(sample_batch.keys())}")
        for key, value in sample_batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
    
    print(f"Final memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024:.2f}GB")
    
except Exception as e:
    print(f"Error during data loading: {str(e)}")
    print("Attempting cleanup...")
    torch.cuda.empty_cache()
    gc.collect()
    
    # Fallback: try with smaller batch size
    print("Trying with smaller batch size...")
    try:
        train_dataloader, test_dataloader, data_loader = create_safe_data_loaders(
            data_path=config_dataset,
            batch_size=64,  # Much smaller batch size
            max_memory_gb=safe_memory_limit
        )
        print("Success with reduced batch size!")
        vmin_all = -1
        vmax_all = 1
    except Exception as e2:
        print(f"Failed even with smaller batch size: {str(e2)}")
        print("Please check available memory and data file integrity.")
        raise

print("\n=== Ready for next steps ===")
print("You can now proceed with the visualization and model training cells.")
print("Note: If you encounter memory issues later, call data_loader.cleanup() to free memory.") 