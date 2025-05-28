# OPTIMIZED FULL DATA PROCESSING PIPELINE
# This replaces multiple cells to prevent memory issues and kernel crashes

import torch
import numpy as np
from typing import Dict, Optional
import gc

def multiply_and_shift_data_optimized(
    data_dict: Dict[str, torch.Tensor], 
    mult_factor: int = 10,
    n_shift: int = 2,
    min_shift_mag: int = -25,
    max_shift_mag: int = 25,
    polarity_change: bool = True,
    chunk_size: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient combined data multiplication and time shifting.
    
    Args:
        data_dict: Input data dictionary
        mult_factor: Data multiplication factor
        n_shift: Number of time shifts per sample
        min_shift_mag: Minimum shift magnitude
        max_shift_mag: Maximum shift magnitude  
        polarity_change: Whether to apply polarity reversal
        chunk_size: Process in chunks to manage memory (None for auto)
    
    Returns:
        Processed data dictionary
    """
    print("Starting optimized data processing...")
    
    # Auto-determine chunk size based on available memory
    if chunk_size is None:
        sample_key = next(key for key in data_dict.keys() if key != 'index')
        data_size = data_dict[sample_key].shape[0]
        # Use smaller chunks for large datasets
        chunk_size = min(50, max(10, data_size // 4))
    
    # Get basic info
    sample_key = next(key for key in data_dict.keys() if key != 'index')
    original_size = data_dict[sample_key].shape[0]
    device = data_dict[sample_key].device
    
    print(f"Original data size: {original_size}")
    print(f"Processing in chunks of: {chunk_size}")
    print(f"Expected final size: {original_size * mult_factor * (n_shift + 1) * (2 if polarity_change else 1)}")
    
    # Calculate filler value from original data (before multiplication)
    filler = torch.mean(data_dict['inputs_embeds'])
    
    # Process in chunks to manage memory
    result_chunks = []
    
    for start_idx in range(0, original_size, chunk_size):
        end_idx = min(start_idx + chunk_size, original_size)
        print(f"Processing chunk {start_idx//chunk_size + 1}/{(original_size + chunk_size - 1)//chunk_size}")
        
        # Extract chunk
        chunk_dict = {}
        for key, tensor in data_dict.items():
            if key == 'index':
                continue
            chunk_dict[key] = tensor[start_idx:end_idx].clone()
        
        # Step 1: Multiply data for current chunk
        chunk_multiplied = multiply_data_chunk(chunk_dict, mult_factor)
        
        # Step 2: Apply time shifts
        chunk_shifted = apply_time_shifts_chunk(
            chunk_multiplied, n_shift, min_shift_mag, max_shift_mag, filler
        )
        
        # Step 3: Apply polarity changes if requested
        if polarity_change:
            chunk_final = apply_polarity_change_chunk(chunk_shifted)
        else:
            chunk_final = chunk_shifted
        
        result_chunks.append(chunk_final)
        
        # Clean up memory
        del chunk_dict, chunk_multiplied, chunk_shifted
        if 'chunk_final' in locals() and polarity_change:
            # chunk_final is different from chunk_shifted only if polarity_change
            pass
        gc.collect()
    
    # Combine all chunks
    print("Combining chunks...")
    final_dict = combine_chunks(result_chunks)
    
    # Add final index
    final_dict['index'] = torch.arange(final_dict[sample_key].shape[0], device=device)
    
    print(f"Final data shape: {final_dict[sample_key].shape}")
    print("Data processing completed successfully!")
    
    return final_dict

def multiply_data_chunk(chunk_dict: Dict[str, torch.Tensor], mult_factor: int) -> Dict[str, torch.Tensor]:
    """Multiply data for a single chunk."""
    result = {}
    for key, tensor in chunk_dict.items():
        if tensor.dim() == 1:
            result[key] = tensor.repeat(mult_factor)
        elif tensor.dim() == 2:
            result[key] = tensor.repeat(mult_factor, 1)
        elif tensor.dim() == 3:
            result[key] = tensor.repeat(mult_factor, 1, 1)
        else:
            repeat_dims = [mult_factor] + [1] * (tensor.dim() - 1)
            result[key] = tensor.repeat(*repeat_dims)
    return result

def apply_time_shifts_chunk(
    chunk_dict: Dict[str, torch.Tensor], 
    n_shift: int, 
    min_shift_mag: int, 
    max_shift_mag: int,
    filler: float
) -> Dict[str, torch.Tensor]:
    """Apply time shifts to a chunk."""
    result_list = [chunk_dict]  # Start with original data
    
    chunk_size = chunk_dict['inputs_embeds'].shape[0]
    
    for n in range(n_shift):
        # Create shifted version
        shifted_chunk = {}
        for key, tensor in chunk_dict.items():
            shifted_chunk[key] = tensor.clone()
        
        # Apply random shifts
        for i in range(chunk_size):
            # Generate non-zero shift
            while True:
                shift_mag = int(torch.randint(low=min_shift_mag-1, high=max_shift_mag+1, size=(1,)))
                if shift_mag != 0:
                    break
            
            # Apply shift to inputs_embeds and labels
            for key in ['inputs_embeds', 'labels']:
                if key in shifted_chunk:
                    shifted_chunk[key][i] = torch.roll(shifted_chunk[key][i], shift_mag, -1)
                    
                    # Fill with filler value
                    if shift_mag > 0:
                        shifted_chunk[key][i, :, :shift_mag] = filler
                    elif shift_mag < 0:
                        shifted_chunk[key][i, :, shifted_chunk[key].shape[-1]+shift_mag:] = filler
        
        result_list.append(shifted_chunk)
    
    # Concatenate all versions
    final_chunk = {}
    for key in chunk_dict.keys():
        tensors_to_cat = [chunk[key] for chunk in result_list]
        final_chunk[key] = torch.cat(tensors_to_cat, dim=0)
    
    return final_chunk

def apply_polarity_change_chunk(chunk_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply polarity reversal to a chunk."""
    # Create polarity-reversed version
    augmented = {}
    for key, tensor in chunk_dict.items():
        if key in ['inputs_embeds', 'labels']:
            augmented[key] = tensor * -1
        else:
            augmented[key] = tensor.clone()
    
    # Combine original and augmented
    final_chunk = {}
    for key in chunk_dict.keys():
        final_chunk[key] = torch.cat([chunk_dict[key], augmented[key]], dim=0)
    
    return final_chunk

def combine_chunks(chunks_list) -> Dict[str, torch.Tensor]:
    """Combine all processed chunks."""
    if not chunks_list:
        raise ValueError("No chunks to combine")
    
    final_dict = {}
    keys = chunks_list[0].keys()
    
    for key in keys:
        tensors_to_cat = [chunk[key] for chunk in chunks_list]
        final_dict[key] = torch.cat(tensors_to_cat, dim=0)
    
    return final_dict

# SIMPLE REPLACEMENT FOR MULTIPLE CELLS
# Replace the data multiplication, time shifts, and polarity reversal cells with this:

# Configuration
mult_factor = 10
n_shift = 2
min_shift_mag = -25
max_shift_mag = 25
polarity_change = True

# Process training data
print("Processing training data...")
snist_train_mlm = multiply_and_shift_data_optimized(
    snist_train_mlm, 
    mult_factor=mult_factor,
    n_shift=n_shift, 
    min_shift_mag=min_shift_mag,
    max_shift_mag=max_shift_mag,
    polarity_change=polarity_change,
    chunk_size=25  # Smaller chunk size for memory safety
)

# Process test data
print("Processing test data...")
snist_test_mlm = multiply_and_shift_data_optimized(
    snist_test_mlm,
    mult_factor=mult_factor, 
    n_shift=n_shift,
    min_shift_mag=min_shift_mag,
    max_shift_mag=max_shift_mag,
    polarity_change=polarity_change,
    chunk_size=25  # Smaller chunk size for memory safety
)

print("All data processing completed!")
print(f"Training data final shape: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Test data final shape: {snist_test_mlm['inputs_embeds'].shape}") 