import torch

def multiply_data_optimized(data_dict, mult_factor=10):
    """
    Optimized function to multiply data N-times efficiently.
    
    Args:
        data_dict: Dictionary containing tensors with keys like 'inputs_embeds', 'labels', etc.
        mult_factor: Multiplication factor (default: 10)
    
    Returns:
        Updated dictionary with multiplied data
    """
    # Get the batch size from any tensor (excluding 'index' which is 1D)
    sample_key = next(key for key in data_dict.keys() if key != 'index')
    batch_size = data_dict[sample_key].shape[0]
    
    # Method 1: Using dictionary comprehension (fastest for most cases)
    updated_dict = {
        key: tensor.repeat(mult_factor, 1, 1) if tensor.dim() == 3 and key != 'index'
        else tensor.repeat(mult_factor) if tensor.dim() == 1 and key != 'index'
        else tensor
        for key, tensor in data_dict.items()
    }
    
    # Update the index to reflect the new size
    updated_dict['index'] = torch.arange(updated_dict[sample_key].shape[0])
    
    return updated_dict

def multiply_data_vectorized(data_dict, mult_factor=10):
    """
    Alternative vectorized approach using torch.repeat_interleave for different patterns.
    
    Args:
        data_dict: Dictionary containing tensors
        mult_factor: Multiplication factor
    
    Returns:
        Updated dictionary with multiplied data
    """
    updated_dict = {}
    sample_key = next(key for key in data_dict.keys() if key != 'index')
    
    for key, tensor in data_dict.items():
        if key == 'index':
            # Skip index, will be updated at the end
            continue
        elif tensor.dim() == 3:
            # For 3D tensors like inputs_embeds, labels, mask_label
            updated_dict[key] = tensor.repeat(mult_factor, 1, 1)
        elif tensor.dim() == 1:
            # For 1D tensors, repeat each element mult_factor times
            updated_dict[key] = tensor.repeat(mult_factor)
        else:
            # For other dimensions, keep as is or handle appropriately
            updated_dict[key] = tensor.repeat(mult_factor, *([1] * (tensor.dim() - 1)))
    
    # Update index
    updated_dict['index'] = torch.arange(updated_dict[sample_key].shape[0])
    
    return updated_dict

def multiply_both_datasets(snist_train_mlm, snist_test_mlm, mult_factor=10):
    """
    Process both training and test datasets in one function call.
    
    Args:
        snist_train_mlm: Training data dictionary
        snist_test_mlm: Test data dictionary  
        mult_factor: Multiplication factor
    
    Returns:
        Tuple of (updated_train_dict, updated_test_dict)
    """
    # Process both datasets
    train_updated = multiply_data_optimized(snist_train_mlm, mult_factor)
    test_updated = multiply_data_optimized(snist_test_mlm, mult_factor)
    
    return train_updated, test_updated

# Usage example for the original code:
"""
# Original slow code:
mult_factor = 10

for key in snist_train_mlm.keys():
    snist_train_mlm[key] = snist_train_mlm[key].repeat(mult_factor, 1, 1)
snist_train_mlm['index'] = torch.arange(snist_train_mlm['inputs_embeds'].shape[0])
    
for key in snist_test_mlm.keys():
    snist_test_mlm[key] = snist_test_mlm[key].repeat(mult_factor, 1, 1)
snist_test_mlm['index'] = torch.arange(snist_test_mlm['inputs_embeds'].shape[0])

# Optimized replacement:
mult_factor = 10
snist_train_mlm, snist_test_mlm = multiply_both_datasets(snist_train_mlm, snist_test_mlm, mult_factor)

# Or process individually:
snist_train_mlm = multiply_data_optimized(snist_train_mlm, mult_factor)
snist_test_mlm = multiply_data_optimized(snist_test_mlm, mult_factor)
"""

def benchmark_methods(data_dict, mult_factor=10, num_runs=5):
    """
    Benchmark different optimization methods.
    
    Args:
        data_dict: Sample data dictionary
        mult_factor: Multiplication factor
        num_runs: Number of runs for timing
    
    Returns:
        Dictionary with timing results
    """
    import time
    
    results = {}
    
    # Method 1: Original approach
    def original_method(data_dict):
        for key in data_dict.keys():
            if key != 'index':
                data_dict[key] = data_dict[key].repeat(mult_factor, 1, 1)
        data_dict['index'] = torch.arange(data_dict['inputs_embeds'].shape[0])
        return data_dict
    
    # Timing original method
    times = []
    for _ in range(num_runs):
        test_dict = {k: v.clone() for k, v in data_dict.items()}
        start = time.time()
        original_method(test_dict)
        times.append(time.time() - start)
    results['original'] = sum(times) / len(times)
    
    # Timing optimized method 1
    times = []
    for _ in range(num_runs):
        test_dict = {k: v.clone() for k, v in data_dict.items()}
        start = time.time()
        multiply_data_optimized(test_dict, mult_factor)
        times.append(time.time() - start)
    results['optimized'] = sum(times) / len(times)
    
    # Timing vectorized method
    times = []
    for _ in range(num_runs):
        test_dict = {k: v.clone() for k, v in data_dict.items()}
        start = time.time()
        multiply_data_vectorized(test_dict, mult_factor)
        times.append(time.time() - start)
    results['vectorized'] = sum(times) / len(times)
    
    return results 