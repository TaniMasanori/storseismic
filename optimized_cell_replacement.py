# Optimized data multiplication cell - replace the original slow code with this

import torch
from typing import Dict

def multiply_data_fast(data_dict: Dict[str, torch.Tensor], mult_factor: int = 10) -> Dict[str, torch.Tensor]:
    """
    Fast multiplication of data N-times with proper dimension handling.
    
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
            
        # Handle different tensor dimensions efficiently
        if tensor.dim() == 1:
            result_dict[key] = tensor.repeat(mult_factor)
        elif tensor.dim() == 2: 
            result_dict[key] = tensor.repeat(mult_factor, 1)
        elif tensor.dim() == 3:
            result_dict[key] = tensor.repeat(mult_factor, 1, 1)
        else:
            # General case for higher dimensions
            repeat_dims = [mult_factor] + [1] * (tensor.dim() - 1)
            result_dict[key] = tensor.repeat(*repeat_dims)
    
    # Create new index with proper device and dtype
    new_size = result_dict[sample_key].shape[0]
    result_dict['index'] = torch.arange(new_size, device=device, dtype=dtype)
    
    return result_dict

# OPTIMIZED REPLACEMENT CODE:
# Replace the original slow loops with this single optimized operation
mult_factor = 10

# Process both datasets efficiently
snist_train_mlm = multiply_data_fast(snist_train_mlm, mult_factor)
snist_test_mlm = multiply_data_fast(snist_test_mlm, mult_factor)

print(f"Data multiplication completed with factor {mult_factor}")
print(f"Training data new shape: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Test data new shape: {snist_test_mlm['inputs_embeds'].shape}") 