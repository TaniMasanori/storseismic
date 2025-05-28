import torch
from typing import Dict, Any, Tuple

def multiply_data_inplace(data_dict: Dict[str, torch.Tensor], mult_factor: int = 10) -> None:
    """
    In-place multiplication of data N-times for memory efficiency.
    Modifies the original dictionary.
    
    Args:
        data_dict: Dictionary containing tensors 
        mult_factor: Multiplication factor
    """
    # Store original index key to handle separately
    original_keys = list(data_dict.keys())
    sample_key = next(key for key in original_keys if key != 'index')
    
    for key in original_keys:
        if key == 'index':
            continue
        
        tensor = data_dict[key]
        if tensor.dim() == 3:
            # For 3D tensors: repeat along the batch dimension
            data_dict[key] = tensor.repeat(mult_factor, 1, 1)
        elif tensor.dim() == 1:
            # For 1D tensors: repeat each element
            data_dict[key] = tensor.repeat(mult_factor)
        elif tensor.dim() == 2:
            # For 2D tensors: repeat along the first dimension
            data_dict[key] = tensor.repeat(mult_factor, 1)
    
    # Update index last
    data_dict['index'] = torch.arange(data_dict[sample_key].shape[0])

def multiply_data_chunked(data_dict: Dict[str, torch.Tensor], 
                         mult_factor: int = 10, 
                         chunk_size: int = 100) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient chunked processing for very large datasets.
    
    Args:
        data_dict: Dictionary containing tensors
        mult_factor: Multiplication factor  
        chunk_size: Process data in chunks of this size
    
    Returns:
        Updated dictionary with multiplied data
    """
    sample_key = next(key for key in data_dict.keys() if key != 'index')
    total_samples = data_dict[sample_key].shape[0]
    
    # Initialize result dictionary
    result_dict = {}
    
    # Process in chunks to manage memory
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        
        # Process current chunk
        chunk_dict = {}
        for key, tensor in data_dict.items():
            if key == 'index':
                continue
            
            chunk = tensor[start_idx:end_idx]
            if tensor.dim() == 3:
                repeated_chunk = chunk.repeat(mult_factor, 1, 1)
            elif tensor.dim() == 1:
                repeated_chunk = chunk.repeat(mult_factor)
            elif tensor.dim() == 2:
                repeated_chunk = chunk.repeat(mult_factor, 1)
            
            if key not in chunk_dict:
                chunk_dict[key] = []
            chunk_dict[key].append(repeated_chunk)
        
        # Concatenate chunks
        for key, chunk_list in chunk_dict.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].extend(chunk_list)
    
    # Final concatenation
    final_dict = {}
    for key, tensor_list in result_dict.items():
        final_dict[key] = torch.cat(tensor_list, dim=0)
    
    # Add index
    final_dict['index'] = torch.arange(final_dict[sample_key].shape[0])
    
    return final_dict

def multiply_data_optimized_v2(data_dict: Dict[str, torch.Tensor], 
                              mult_factor: int = 10) -> Dict[str, torch.Tensor]:
    """
    Enhanced optimized version with better error handling and type checking.
    
    Args:
        data_dict: Dictionary containing tensors
        mult_factor: Multiplication factor
    
    Returns:
        Updated dictionary with multiplied data
    """
    if mult_factor <= 0:
        raise ValueError("mult_factor must be positive")
    
    if mult_factor == 1:
        # No multiplication needed, just update index
        result = data_dict.copy()
        sample_key = next(key for key in result.keys() if key != 'index')
        result['index'] = torch.arange(result[sample_key].shape[0])
        return result
    
    # Get device and dtype from first tensor for consistency
    sample_key = next(key for key in data_dict.keys() if key != 'index')
    sample_tensor = data_dict[sample_key]
    device = sample_tensor.device
    dtype = sample_tensor.dtype if sample_tensor.dtype != torch.bool else torch.int64
    
    result_dict = {}
    
    for key, tensor in data_dict.items():
        if key == 'index':
            continue
            
        # Ensure tensor is on the same device
        if tensor.device != device:
            tensor = tensor.to(device)
        
        # Handle different tensor dimensions
        if tensor.dim() == 1:
            result_dict[key] = tensor.repeat(mult_factor)
        elif tensor.dim() == 2: 
            result_dict[key] = tensor.repeat(mult_factor, 1)
        elif tensor.dim() == 3:
            result_dict[key] = tensor.repeat(mult_factor, 1, 1)
        elif tensor.dim() == 4:
            result_dict[key] = tensor.repeat(mult_factor, 1, 1, 1)
        else:
            # General case for higher dimensions
            repeat_dims = [mult_factor] + [1] * (tensor.dim() - 1)
            result_dict[key] = tensor.repeat(*repeat_dims)
    
    # Create new index with proper device and dtype
    new_size = result_dict[sample_key].shape[0]
    result_dict['index'] = torch.arange(new_size, device=device, dtype=dtype)
    
    return result_dict

def process_datasets_parallel(snist_train_mlm: Dict[str, torch.Tensor], 
                            snist_test_mlm: Dict[str, torch.Tensor], 
                            mult_factor: int = 10,
                            use_chunked: bool = False,
                            chunk_size: int = 100) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Process both datasets with option for parallel or chunked processing.
    
    Args:
        snist_train_mlm: Training data dictionary
        snist_test_mlm: Test data dictionary
        mult_factor: Multiplication factor
        use_chunked: Whether to use chunked processing
        chunk_size: Chunk size for chunked processing
    
    Returns:
        Tuple of (processed_train_dict, processed_test_dict)
    """
    if use_chunked:
        train_result = multiply_data_chunked(snist_train_mlm, mult_factor, chunk_size)
        test_result = multiply_data_chunked(snist_test_mlm, mult_factor, chunk_size)
    else:
        train_result = multiply_data_optimized_v2(snist_train_mlm, mult_factor)
        test_result = multiply_data_optimized_v2(snist_test_mlm, mult_factor)
    
    return train_result, test_result

# Usage examples and comparison
"""
# Original code (SLOW):
mult_factor = 10

for key in snist_train_mlm.keys():
    snist_train_mlm[key] = snist_train_mlm[key].repeat(mult_factor, 1, 1)
snist_train_mlm['index'] = torch.arange(snist_train_mlm['inputs_embeds'].shape[0])
    
for key in snist_test_mlm.keys():
    snist_test_mlm[key] = snist_test_mlm[key].repeat(mult_factor, 1, 1)
snist_test_mlm['index'] = torch.arange(snist_test_mlm['inputs_embeds'].shape[0])

# OPTIMIZED REPLACEMENTS:

# Option 1: Fast in-place modification (modifies original dictionaries)
mult_factor = 10
multiply_data_inplace(snist_train_mlm, mult_factor)
multiply_data_inplace(snist_test_mlm, mult_factor)

# Option 2: Fast with new dictionaries (preserves originals)
mult_factor = 10
snist_train_mlm = multiply_data_optimized_v2(snist_train_mlm, mult_factor)
snist_test_mlm = multiply_data_optimized_v2(snist_test_mlm, mult_factor)

# Option 3: Process both at once
mult_factor = 10
snist_train_mlm, snist_test_mlm = process_datasets_parallel(
    snist_train_mlm, snist_test_mlm, mult_factor
)

# Option 4: For very large datasets, use chunked processing
mult_factor = 10
snist_train_mlm, snist_test_mlm = process_datasets_parallel(
    snist_train_mlm, snist_test_mlm, mult_factor, 
    use_chunked=True, chunk_size=50
)
""" 