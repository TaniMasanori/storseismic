# SAFE DATA PROCESSING CELL - PREVENTS KERNEL CRASHES
# Replace multiple problematic cells with this safe version

import torch
import gc

def safe_data_processing(data_dict, mult_factor=10, enable_shifts=True, enable_polarity=True):
    """
    Safe data processing with memory management to prevent kernel crashes.
    
    Args:
        data_dict: Input data dictionary
        mult_factor: Multiplication factor for data augmentation
        enable_shifts: Whether to apply time shifts (can be disabled to save memory)
        enable_polarity: Whether to apply polarity reversal
    
    Returns:
        Processed data dictionary
    """
    print(f"Starting safe data processing...")
    print(f"Original shape: {data_dict['inputs_embeds'].shape}")
    
    # Step 1: Data multiplication (optimized)
    print("Step 1: Data multiplication...")
    processed_data = {}
    
    for key, tensor in data_dict.items():
        if key == 'index':
            continue
        elif tensor.dim() == 3:
            processed_data[key] = tensor.repeat(mult_factor, 1, 1)
        elif tensor.dim() == 1:
            processed_data[key] = tensor.repeat(mult_factor)
        elif tensor.dim() == 2:
            processed_data[key] = tensor.repeat(mult_factor, 1)
    
    # Update index
    processed_data['index'] = torch.arange(processed_data['inputs_embeds'].shape[0])
    
    print(f"After multiplication: {processed_data['inputs_embeds'].shape}")
    gc.collect()  # Free memory
    
    # Step 2: Time shifts (optional, memory-intensive)
    if enable_shifts:
        print("Step 2: Applying time shifts...")
        
        # Use smaller shifts to reduce memory usage
        n_shift = 1  # Reduced from 2 to 1 to save memory
        min_shift_mag = -15  # Reduced range
        max_shift_mag = 15   # Reduced range
        
        filler = torch.mean(processed_data['inputs_embeds'])
        data_len = processed_data['inputs_embeds'].shape[0]
        
        shifted_versions = []
        
        for n in range(n_shift):
            print(f"  Creating shift version {n+1}/{n_shift}...")
            
            # Create shifted copy
            data2 = {}
            for key, value in processed_data.items():
                if key != 'index':
                    data2[key] = value.clone()
            
            # Apply shifts in smaller batches
            batch_size = 50  # Process in smaller batches
            for start_idx in range(0, data_len, batch_size):
                end_idx = min(start_idx + batch_size, data_len)
                
                for i in range(start_idx, end_idx):
                    # Generate non-zero shift
                    shift_mag = 0
                    while shift_mag == 0:
                        shift_mag = int(torch.randint(low=min_shift_mag-1, high=max_shift_mag+1, size=(1,)))
                    
                    # Apply shift
                    data2['inputs_embeds'][i] = torch.roll(data2['inputs_embeds'][i], shift_mag, -1)
                    data2['labels'][i] = torch.roll(data2['labels'][i], shift_mag, -1)
                    
                    # Fill edges
                    if shift_mag > 0:
                        data2['inputs_embeds'][i, :, :shift_mag] = filler
                        data2['labels'][i, :, :shift_mag] = filler
                    elif shift_mag < 0:
                        data2['inputs_embeds'][i, :, data2['inputs_embeds'].shape[-1]+shift_mag:] = filler
                        data2['labels'][i, :, data2['labels'].shape[-1]+shift_mag:] = filler
            
            shifted_versions.append(data2)
            gc.collect()  # Free memory after each shift
        
        # Combine all versions
        print("  Combining shifted versions...")
        for data2 in shifted_versions:
            for key in processed_data.keys():
                if key != 'index':
                    processed_data[key] = torch.cat((processed_data[key], data2[key]), 0)
        
        # Update index
        processed_data['index'] = torch.arange(processed_data['inputs_embeds'].shape[0])
        
        print(f"After time shifts: {processed_data['inputs_embeds'].shape}")
        gc.collect()
    else:
        print("Step 2: Skipping time shifts (disabled for memory safety)")
    
    # Step 3: Polarity reversal (optional)
    if enable_polarity:
        print("Step 3: Applying polarity reversal...")
        
        # Create augmented version
        augmented = {}
        for key, tensor in processed_data.items():
            if key in ['inputs_embeds', 'labels']:
                augmented[key] = tensor * -1
            else:
                augmented[key] = tensor.clone()
        
        # Combine original and augmented
        for key in processed_data.keys():
            if key != 'index':
                processed_data[key] = torch.cat((processed_data[key], augmented[key]), 0)
        
        # Update index
        processed_data['index'] = torch.arange(processed_data['inputs_embeds'].shape[0])
        
        print(f"After polarity reversal: {processed_data['inputs_embeds'].shape}")
        gc.collect()
    else:
        print("Step 3: Skipping polarity reversal (disabled)")
    
    print("Safe data processing completed!")
    return processed_data

# =============================================================================
# USAGE - REPLACE MULTIPLE CELLS WITH THIS SINGLE CELL
# =============================================================================

# Configuration - adjust these parameters based on your memory constraints
mult_factor = 10

# For systems with limited memory, set enable_shifts=False or enable_polarity=False
enable_shifts = True      # Set to False if you get memory errors
enable_polarity = True    # Set to False if you get memory errors

# Process training data
print("=" * 50)
print("PROCESSING TRAINING DATA")
print("=" * 50)
snist_train_mlm = safe_data_processing(
    snist_train_mlm, 
    mult_factor=mult_factor,
    enable_shifts=enable_shifts,
    enable_polarity=enable_polarity
)

# Process test data  
print("=" * 50)
print("PROCESSING TEST DATA")
print("=" * 50)
snist_test_mlm = safe_data_processing(
    snist_test_mlm,
    mult_factor=mult_factor,
    enable_shifts=enable_shifts, 
    enable_polarity=enable_polarity
)

print("=" * 50)
print("ALL PROCESSING COMPLETED!")
print(f"Final training shape: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Final test shape: {snist_test_mlm['inputs_embeds'].shape}")
print("=" * 50)

# Memory cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache cleared") 