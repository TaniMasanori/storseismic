# ULTRA CONSERVATIVE CELL - LAST RESORT
# If this still crashes, there may be a deeper system issue

import torch
import gc
import time

print("Starting ultra-conservative data processing...")

# STEP 1: Check current data state
print("Current data shapes:")
print(f"Training: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Test: {snist_test_mlm['inputs_embeds'].shape}")

# STEP 2: Ultra-conservative multiplication function
def ultra_safe_multiply(data_dict, mult_factor=10):
    """
    Ultra-conservative data multiplication with maximum safety measures.
    """
    print(f"\nProcessing data with shape: {data_dict['inputs_embeds'].shape}")
    
    # Get original size
    original_size = data_dict['inputs_embeds'].shape[0]
    
    # Create empty result tensors first
    result = {}
    
    # Calculate target shapes
    for key, tensor in data_dict.items():
        if key == 'index':
            continue
        
        if tensor.dim() == 3:
            target_shape = (original_size * mult_factor, tensor.shape[1], tensor.shape[2])
        elif tensor.dim() == 2:
            target_shape = (original_size * mult_factor, tensor.shape[1])
        elif tensor.dim() == 1:
            target_shape = (original_size * mult_factor,)
        
        print(f"Creating {key} with target shape: {target_shape}")
        
        # Create empty tensor
        result[key] = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Fill in chunks of 1 sample at a time
        for i in range(original_size):
            if i % 50 == 0:  # Progress every 50 samples
                print(f"  Processing sample {i}/{original_size}")
                gc.collect()
                time.sleep(0.01)  # Small pause
            
            # Copy this sample mult_factor times
            for j in range(mult_factor):
                target_idx = i * mult_factor + j
                result[key][target_idx] = tensor[i]
        
        print(f"Completed {key}")
        gc.collect()
    
    # Create index
    result['index'] = torch.arange(result['inputs_embeds'].shape[0])
    
    print(f"Final shape: {result['inputs_embeds'].shape}")
    return result

# STEP 3: Try with reduced multiplication factor first
print("\n" + "="*50)
print("TESTING WITH REDUCED MULTIPLICATION FACTOR")
print("="*50)

# Start with smaller multiplication factor
test_mult_factor = 2  # Start small

try:
    print("Testing with training data (mult_factor=2)...")
    test_result = ultra_safe_multiply(snist_train_mlm, test_mult_factor)
    print("✓ Test successful with mult_factor=2")
    
    # Clean up test
    del test_result
    gc.collect()
    
    # Now try with full factor
    print("\nProceeding with full multiplication factor...")
    mult_factor = 10
    
except Exception as e:
    print(f"✗ Even mult_factor=2 failed: {e}")
    print("Trying mult_factor=1 (no multiplication)...")
    mult_factor = 1

# STEP 4: Process with determined multiplication factor
print(f"\nUsing multiplication factor: {mult_factor}")

# Process training data
print("\n" + "="*50)
print("PROCESSING TRAINING DATA")
print("="*50)

try:
    snist_train_mlm = ultra_safe_multiply(snist_train_mlm, mult_factor)
    print("✓ Training data processed successfully")
except Exception as e:
    print(f"✗ Training data failed: {e}")
    print("Keeping original training data unchanged")

# Aggressive cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
time.sleep(2)  # Longer pause

# Process test data
print("\n" + "="*50)
print("PROCESSING TEST DATA")
print("="*50)

try:
    snist_test_mlm = ultra_safe_multiply(snist_test_mlm, mult_factor)
    print("✓ Test data processed successfully")
except Exception as e:
    print(f"✗ Test data failed: {e}")
    print("Keeping original test data unchanged")

# Final cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*50)
print("ULTRA-CONSERVATIVE PROCESSING COMPLETED")
print(f"Final training shape: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Final test shape: {snist_test_mlm['inputs_embeds'].shape}")
print("="*50)

# STEP 5: Alternative approach if everything fails
print("\nIf the above still crashes, try this manual approach:")
print("""
# MANUAL APPROACH - Run each line separately in different cells:

# Cell 1:
mult_factor = 2  # Start with smaller factor
print("Starting manual processing...")

# Cell 2:
train_inputs = snist_train_mlm['inputs_embeds'].repeat(mult_factor, 1, 1)
print("Training inputs done")

# Cell 3:
train_labels = snist_train_mlm['labels'].repeat(mult_factor, 1, 1)
print("Training labels done")

# Cell 4:
train_mask = snist_train_mlm['mask_label'].repeat(mult_factor, 1, 1)
print("Training mask done")

# Cell 5:
snist_train_mlm = {
    'inputs_embeds': train_inputs,
    'labels': train_labels, 
    'mask_label': train_mask,
    'index': torch.arange(train_inputs.shape[0])
}
print("Training data reconstructed")

# Repeat similar process for test data...
""") 