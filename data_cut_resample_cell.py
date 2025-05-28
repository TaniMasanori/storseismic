# =============================================================================
# DATA CUTTING AND RESAMPLING CELL
# =============================================================================
# This cell cuts data after 1024 samples and resamples to 1/4 (256 samples)
# for time domain processing. Insert this cell into your notebook after the
# data loading and before augmentation steps.

import torch
import torch.nn.functional as F

def cut_and_resample_data(data_dict, cut_length=1024, target_samples=256):
    """
    Cut data after specified length and resample to target number of samples.
    
    Args:
        data_dict (dict): Dictionary containing seismic data tensors
        cut_length (int): Number of samples to keep from original data (default: 1024)
        target_samples (int): Target number of samples after resampling (default: 256)
        
    Returns:
        dict: Dictionary with processed data tensors
    """
    print(f"Starting data cutting and resampling...")
    print(f"Cut length: {cut_length}, Target samples: {target_samples}")
    
    processed_data = {}
    
    for key, tensor in data_dict.items():
        if key == 'index':
            # Keep index unchanged
            processed_data[key] = tensor.clone()
            continue
            
        print(f"Processing {key} with original shape: {tensor.shape}")
        
        # Clone the tensor to avoid modifying the original
        processed_tensor = tensor.clone()
        
        # Get the time dimension (assuming it's the last dimension)
        original_time_samples = processed_tensor.shape[-1]
        
        # Step 1: Cut data after cut_length samples
        if original_time_samples > cut_length:
            print(f"  Cutting {key} from {original_time_samples} to {cut_length} samples")
            processed_tensor = processed_tensor[..., :cut_length]
        else:
            print(f"  Warning: {key} has only {original_time_samples} samples, no cutting needed")
        
        # Step 2: Resample to target_samples (1/4 downsampling)
        current_samples = processed_tensor.shape[-1]
        
        if current_samples != target_samples:
            print(f"  Resampling {key} from {current_samples} to {target_samples} samples")
            
            # Use interpolation for resampling
            # Reshape for interpolation: (batch_size * channels, 1, time_samples)
            original_shape = processed_tensor.shape
            
            if len(original_shape) == 3:  # (batch, channels, time)
                batch_size, channels, time_samples = original_shape
                reshaped = processed_tensor.view(batch_size * channels, 1, time_samples)
            elif len(original_shape) == 2:  # (batch, time) 
                batch_size, time_samples = original_shape
                channels = 1
                reshaped = processed_tensor.view(batch_size, 1, time_samples)
            else:
                raise ValueError(f"Unsupported tensor shape: {original_shape}")
            
            # Interpolate to target samples
            resampled = F.interpolate(
                reshaped.float(), 
                size=target_samples, 
                mode='linear', 
                align_corners=True
            )
            
            # Reshape back to original format
            if len(original_shape) == 3:
                processed_tensor = resampled.view(batch_size, channels, target_samples)
            else:
                processed_tensor = resampled.view(batch_size, target_samples)
                
            # Convert back to original dtype if needed
            processed_tensor = processed_tensor.to(tensor.dtype)
        
        processed_data[key] = processed_tensor
        print(f"  Finished processing {key}, new shape: {processed_tensor.shape}")
    
    print("Data cutting and resampling completed!")
    return processed_data

# =============================================================================
# APPLY CUTTING AND RESAMPLING
# =============================================================================

print("=" * 80)
print("APPLYING DATA CUTTING AND RESAMPLING")
print("=" * 80)
print("This will cut data after 1024 samples and resample to 256 samples")
print("Original time dimension will change from current size to 256")
print()

# Check current shapes
print("Current data shapes:")
print(f"Training data: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Testing data: {snist_test_mlm['inputs_embeds'].shape}")
print()

# Apply cutting and resampling to training data
print("Processing training data...")
snist_train_mlm = cut_and_resample_data(snist_train_mlm, cut_length=1024, target_samples=256)
print()

# Apply cutting and resampling to testing data  
print("Processing testing data...")
snist_test_mlm = cut_and_resample_data(snist_test_mlm, cut_length=1024, target_samples=256)
print()

# Show final shapes
print("Final data shapes:")
print(f"Training data: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Testing data: {snist_test_mlm['inputs_embeds'].shape}")
print()

print("=" * 80)
print("DATA CUTTING AND RESAMPLING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("You can now proceed with the rest of your data processing pipeline.")
print("The time dimension has been reduced from the original size to 256 samples.") 