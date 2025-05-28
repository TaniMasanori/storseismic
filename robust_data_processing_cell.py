# ROBUST DATA PROCESSING - PREVENTS KERNEL CRASHES
# Even when RAM is available, this handles PyTorch memory management issues

import torch
import gc
import time
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def force_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(0.1)  # Give system time to clean up

def robust_data_processing(data_dict, mult_factor=10, enable_shifts=True, enable_polarity=True):
    """
    Robust data processing that prevents kernel crashes through careful memory management.
    
    Args:
        data_dict: Input data dictionary
        mult_factor: Multiplication factor for data augmentation
        enable_shifts: Whether to apply time shifts
        enable_polarity: Whether to apply polarity reversal
    
    Returns:
        Processed data dictionary
    """
    print(f"Starting robust data processing...")
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    print(f"Original shape: {data_dict['inputs_embeds'].shape}")
    
    # Create a copy to avoid modifying original data
    working_data = {}
    for key, tensor in data_dict.items():
        working_data[key] = tensor.detach().clone()
    
    force_cleanup()
    
    # Step 1: Data multiplication with memory monitoring
    print("\nStep 1: Data multiplication...")
    print(f"Memory before multiplication: {get_memory_usage():.1f} MB")
    
    processed_data = {}
    
    # Process each key separately to avoid large temporary objects
    for key, tensor in working_data.items():
        if key == 'index':
            continue
        
        print(f"  Processing key: {key}")
        
        # Handle different tensor dimensions
        if tensor.dim() == 3:
            processed_data[key] = tensor.repeat(mult_factor, 1, 1)
        elif tensor.dim() == 1:
            processed_data[key] = tensor.repeat(mult_factor)
        elif tensor.dim() == 2:
            processed_data[key] = tensor.repeat(mult_factor, 1)
        
        # Force cleanup after each key
        del tensor
        force_cleanup()
    
    # Update index
    processed_data['index'] = torch.arange(processed_data['inputs_embeds'].shape[0])
    
    # Clean up working data
    del working_data
    force_cleanup()
    
    print(f"After multiplication: {processed_data['inputs_embeds'].shape}")
    print(f"Memory after multiplication: {get_memory_usage():.1f} MB")
    
    # Step 2: Time shifts with ultra-conservative memory management
    if enable_shifts:
        print("\nStep 2: Applying time shifts...")
        print(f"Memory before shifts: {get_memory_usage():.1f} MB")
        
        # Reduce parameters to be more conservative
        n_shift = 1  # Only 1 shift instead of 2
        min_shift_mag = -10  # Smaller range
        max_shift_mag = 10   # Smaller range
        
        filler = torch.mean(processed_data['inputs_embeds']).item()  # Convert to scalar
        data_len = processed_data['inputs_embeds'].shape[0]
        
        print(f"  Applying {n_shift} time shift(s) to {data_len} samples")
        
        # Process shifts one at a time with aggressive cleanup
        for n in range(n_shift):
            print(f"  Creating shift version {n+1}/{n_shift}...")
            print(f"  Memory before shift {n+1}: {get_memory_usage():.1f} MB")
            
            # Create shifted copy with minimal memory footprint
            data2 = {}
            
            # Copy each tensor separately
            for key in ['inputs_embeds', 'labels', 'mask_label']:
                if key in processed_data:
                    data2[key] = processed_data[key].clone()
                    force_cleanup()
            
            # Apply shifts in very small batches
            batch_size = 25  # Even smaller batches
            
            for start_idx in range(0, data_len, batch_size):
                end_idx = min(start_idx + batch_size, data_len)
                
                if start_idx % (batch_size * 10) == 0:  # Progress every 10 batches
                    print(f"    Processing samples {start_idx}-{end_idx-1}/{data_len}")
                    print(f"    Memory: {get_memory_usage():.1f} MB")
                
                for i in range(start_idx, end_idx):
                    # Generate non-zero shift
                    shift_mag = 0
                    attempts = 0
                    while shift_mag == 0 and attempts < 10:
                        shift_mag = int(torch.randint(low=min_shift_mag-1, high=max_shift_mag+1, size=(1,)).item())
                        attempts += 1
                    
                    if shift_mag == 0:  # Fallback
                        shift_mag = 1
                    
                    # Apply shift to inputs_embeds and labels only
                    for key in ['inputs_embeds', 'labels']:
                        if key in data2:
                            # Use in-place operations where possible
                            data2[key][i] = torch.roll(data2[key][i], shift_mag, -1)
                            
                            # Fill edges
                            if shift_mag > 0:
                                data2[key][i, :, :shift_mag] = filler
                            elif shift_mag < 0:
                                data2[key][i, :, data2[key].shape[-1]+shift_mag:] = filler
                
                # Cleanup every batch
                if start_idx % (batch_size * 5) == 0:
                    force_cleanup()
            
            print(f"  Memory after creating shift {n+1}: {get_memory_usage():.1f} MB")
            
            # Combine with original data
            print(f"  Combining shift {n+1} with original data...")
            
            for key in processed_data.keys():
                if key != 'index' and key in data2:
                    # Concatenate and immediately replace
                    temp = torch.cat((processed_data[key], data2[key]), 0)
                    del processed_data[key]
                    processed_data[key] = temp
                    del temp
                    force_cleanup()
            
            # Clean up shifted data
            del data2
            force_cleanup()
            
            print(f"  Memory after combining shift {n+1}: {get_memory_usage():.1f} MB")
        
        # Update index
        processed_data['index'] = torch.arange(processed_data['inputs_embeds'].shape[0])
        
        print(f"After time shifts: {processed_data['inputs_embeds'].shape}")
        print(f"Memory after all shifts: {get_memory_usage():.1f} MB")
    else:
        print("\nStep 2: Skipping time shifts (disabled for memory safety)")
    
    # Step 3: Polarity reversal with careful memory management
    if enable_polarity:
        print("\nStep 3: Applying polarity reversal...")
        print(f"Memory before polarity: {get_memory_usage():.1f} MB")
        
        # Create augmented version key by key
        augmented = {}
        
        for key in processed_data.keys():
            if key == 'index':
                augmented[key] = processed_data[key].clone()
            elif key in ['inputs_embeds', 'labels']:
                print(f"  Creating polarity-reversed {key}")
                augmented[key] = processed_data[key] * -1
                force_cleanup()
            else:
                augmented[key] = processed_data[key].clone()
                force_cleanup()
        
        print(f"Memory after creating augmented data: {get_memory_usage():.1f} MB")
        
        # Combine original and augmented key by key
        for key in processed_data.keys():
            if key != 'index':
                print(f"  Combining {key}")
                temp = torch.cat([processed_data[key], augmented[key]], dim=0)
                del processed_data[key]
                del augmented[key]
                processed_data[key] = temp
                del temp
                force_cleanup()
        
        # Clean up augmented data
        del augmented
        force_cleanup()
        
        # Update index
        processed_data['index'] = torch.arange(processed_data['inputs_embeds'].shape[0])
        
        print(f"After polarity reversal: {processed_data['inputs_embeds'].shape}")
        print(f"Memory after polarity: {get_memory_usage():.1f} MB")
    else:
        print("\nStep 3: Skipping polarity reversal (disabled)")
    
    print(f"\nRobust data processing completed!")
    print(f"Final memory usage: {get_memory_usage():.1f} MB")
    return processed_data

# =============================================================================
# USAGE - ULTRA-SAFE VERSION
# =============================================================================

# Check if psutil is available
try:
    import psutil
    psutil_available = True
except ImportError:
    print("Warning: psutil not available. Install with: pip install psutil")
    psutil_available = False
    
    # Fallback function
    def get_memory_usage():
        return 0.0

# Configuration
mult_factor = 10

# Conservative settings to prevent crashes
enable_shifts = True      # Can be set to False if still having issues
enable_polarity = True    # Can be set to False if still having issues

print("=" * 60)
print("ROBUST DATA PROCESSING - CRASH PREVENTION")
print("=" * 60)

# Process training data
print("\nPROCESSING TRAINING DATA")
print("-" * 30)
snist_train_mlm = robust_data_processing(
    snist_train_mlm, 
    mult_factor=mult_factor,
    enable_shifts=enable_shifts,
    enable_polarity=enable_polarity
)

# Aggressive cleanup between datasets
force_cleanup()
time.sleep(1)  # Give system time to stabilize

# Process test data  
print("\nPROCESSING TEST DATA")
print("-" * 30)
snist_test_mlm = robust_data_processing(
    snist_test_mlm,
    mult_factor=mult_factor,
    enable_shifts=enable_shifts, 
    enable_polarity=enable_polarity
)

print("\n" + "=" * 60)
print("ALL PROCESSING COMPLETED SUCCESSFULLY!")
print(f"Final training shape: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Final test shape: {snist_test_mlm['inputs_embeds'].shape}")
print(f"Final memory usage: {get_memory_usage():.1f} MB")
print("=" * 60)

# Final cleanup
force_cleanup()
print("Memory cleanup completed") 