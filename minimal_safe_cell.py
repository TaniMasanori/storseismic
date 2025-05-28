# MINIMAL SAFE CELL - ONLY DATA MULTIPLICATION
# Use this if you're still getting crashes with the other versions

import torch
import gc

print("Starting minimal safe data processing...")

# Configuration
mult_factor = 10

# Function for safe data multiplication only
def minimal_multiply_data(data_dict, mult_factor):
    """
    Minimal data multiplication - only the essential operation.
    """
    print(f"Original shape: {data_dict['inputs_embeds'].shape}")
    
    # Create result dictionary
    result = {}
    
    # Process each tensor individually with immediate cleanup
    for key, tensor in data_dict.items():
        if key == 'index':
            continue
        
        print(f"Processing {key}...")
        
        # Handle different dimensions
        if tensor.dim() == 3:
            result[key] = tensor.repeat(mult_factor, 1, 1)
        elif tensor.dim() == 1:
            result[key] = tensor.repeat(mult_factor)
        elif tensor.dim() == 2:
            result[key] = tensor.repeat(mult_factor, 1)
        
        # Immediate cleanup
        gc.collect()
    
    # Add index
    result['index'] = torch.arange(result['inputs_embeds'].shape[0])
    
    print(f"Final shape: {result['inputs_embeds'].shape}")
    return result

# Process training data
print("\n" + "="*40)
print("PROCESSING TRAINING DATA")
print("="*40)
snist_train_mlm = minimal_multiply_data(snist_train_mlm, mult_factor)

# Cleanup between datasets
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Process test data
print("\n" + "="*40)
print("PROCESSING TEST DATA") 
print("="*40)
snist_test_mlm = minimal_multiply_data(snist_test_mlm, mult_factor)

# Final cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*40)
print("MINIMAL PROCESSING COMPLETED!")
print(f"Training shape: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Test shape: {snist_test_mlm['inputs_embeds'].shape}")
print("="*40)

# NOTE: This version only does data multiplication (10x increase)
# If you need time shifts and polarity reversal, run them in separate cells later
# or use the robust version if your system can handle it 