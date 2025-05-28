# Emergency Memory Fix - Quick Solution
# Run this immediately if you're experiencing OOM errors

import torch
import gc
import os

print("=== Emergency GPU Memory Fix ===")

# 1. Clear all GPU memory
def emergency_memory_clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("✓ GPU memory cleared")

# 2. Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 3. Execute memory clear
emergency_memory_clear()

# 4. Check current memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated
    
    print(f"Memory status:")
    print(f"  Total: {total:.2f} GB")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Free: {free:.2f} GB")
    
    if allocated > total * 0.9:
        print("⚠ WARNING: Memory usage is very high!")
        print("Recommended actions:")
        print("1. Restart the kernel")
        print("2. Use smaller batch size")
        print("3. Use gradient accumulation")

# 5. Set recommended batch size based on available memory
recommended_batch_size = max(1, int(free / 2.0))  # Very conservative estimate
print(f"Recommended max batch size: {recommended_batch_size}")

# 6. Quick training settings for memory efficiency
print("\nQuick memory-efficient settings:")
print("batch_size = 1  # Start with smallest possible")
print("accumulation_steps = 16  # Simulate larger batches")
print("torch.cuda.empty_cache()  # Clear after each epoch") 