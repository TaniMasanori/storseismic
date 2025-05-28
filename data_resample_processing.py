"""
Data Resampling and Processing Functions for StorSeismic

This module provides functions to cut seismic data after 1024 samples
and resample to 1/4 (256 samples) for time domain processing.
"""

import torch
import torch.nn.functional as F
import numpy as np


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


def validate_resampled_data(original_data, resampled_data):
    """
    Validate that the resampling process worked correctly.
    
    Args:
        original_data (dict): Original data dictionary
        resampled_data (dict): Resampled data dictionary
    """
    print("\nValidating resampled data...")
    
    for key in original_data.keys():
        if key == 'index':
            continue
            
        orig_shape = original_data[key].shape
        new_shape = resampled_data[key].shape
        
        print(f"{key}:")
        print(f"  Original shape: {orig_shape}")
        print(f"  Resampled shape: {new_shape}")
        
        # Check that only the time dimension changed
        if len(orig_shape) == 3:
            assert orig_shape[0] == new_shape[0], f"Batch size changed for {key}"
            assert orig_shape[1] == new_shape[1], f"Channel size changed for {key}"
            print(f"  Time dimension: {orig_shape[2]} -> {new_shape[2]}")
        elif len(orig_shape) == 2:
            assert orig_shape[0] == new_shape[0], f"Batch size changed for {key}"
            print(f"  Time dimension: {orig_shape[1]} -> {new_shape[1]}")
    
    print("Validation completed successfully!")


def apply_cut_and_resample_to_datasets(train_data_dict, test_data_dict, 
                                     cut_length=1024, target_samples=256):
    """
    Apply cutting and resampling to both training and testing datasets.
    
    Args:
        train_data_dict (dict): Training data dictionary
        test_data_dict (dict): Testing data dictionary
        cut_length (int): Number of samples to keep from original data
        target_samples (int): Target number of samples after resampling
        
    Returns:
        tuple: (processed_train_data, processed_test_data)
    """
    print("=" * 60)
    print("PROCESSING TRAINING DATA")
    print("=" * 60)
    processed_train = cut_and_resample_data(train_data_dict, cut_length, target_samples)
    
    print("\n" + "=" * 60)
    print("PROCESSING TESTING DATA")
    print("=" * 60)
    processed_test = cut_and_resample_data(test_data_dict, cut_length, target_samples)
    
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print("Training data validation:")
    validate_resampled_data(train_data_dict, processed_train)
    
    print("\nTesting data validation:")
    validate_resampled_data(test_data_dict, processed_test)
    
    return processed_train, processed_test


# Example usage function
def example_usage():
    """
    Example of how to use the cutting and resampling functions.
    """
    print("Example usage of data cutting and resampling functions:")
    print("""
    # Apply to your data dictionaries:
    processed_train, processed_test = apply_cut_and_resample_to_datasets(
        snist_train_mlm, 
        snist_test_mlm, 
        cut_length=1024, 
        target_samples=256
    )
    
    # Update your data dictionaries:
    snist_train_mlm = processed_train
    snist_test_mlm = processed_test
    """)


if __name__ == "__main__":
    example_usage() 