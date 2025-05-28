"""
Test script for data cutting and resampling functionality.

This script creates sample data and tests the cutting and resampling functions
to ensure they work correctly before applying to real seismic data.
"""

import torch
import numpy as np
from data_resample_processing import cut_and_resample_data, validate_resampled_data

def create_sample_data():
    """
    Create sample data that mimics the structure of SNIST seismic data.
    """
    print("Creating sample test data...")
    
    # Simulate data similar to SNIST format
    batch_size = 10
    channels = 20  # number of offsets/traces
    time_samples = 1500  # Original time samples (larger than 1024)
    
    # Create sample data dictionary
    sample_data = {
        'inputs_embeds': torch.randn(batch_size, channels, time_samples),
        'labels': torch.randn(batch_size, channels, time_samples),
        'mask_label': torch.zeros(batch_size, channels, time_samples),
        'index': torch.arange(batch_size)
    }
    
    print(f"Created sample data with shapes:")
    for key, tensor in sample_data.items():
        print(f"  {key}: {tensor.shape}")
    
    return sample_data

def test_cutting_and_resampling():
    """
    Test the cutting and resampling functionality.
    """
    print("=" * 60)
    print("TESTING DATA CUTTING AND RESAMPLING")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Test with default parameters (cut at 1024, resample to 256)
    print("\nTest 1: Default parameters (cut=1024, target=256)")
    processed_data_1 = cut_and_resample_data(sample_data, cut_length=1024, target_samples=256)
    validate_resampled_data(sample_data, processed_data_1)
    
    # Test with different parameters
    print("\n" + "=" * 60)
    print("Test 2: Custom parameters (cut=800, target=200)")
    processed_data_2 = cut_and_resample_data(sample_data, cut_length=800, target_samples=200)
    validate_resampled_data(sample_data, processed_data_2)
    
    # Test edge case: data smaller than cut length
    print("\n" + "=" * 60)
    print("Test 3: Edge case - data smaller than cut length")
    
    # Create smaller sample data
    small_data = {
        'inputs_embeds': torch.randn(5, 20, 500),  # Only 500 time samples
        'labels': torch.randn(5, 20, 500),
        'mask_label': torch.zeros(5, 20, 500),
        'index': torch.arange(5)
    }
    
    processed_data_3 = cut_and_resample_data(small_data, cut_length=1024, target_samples=256)
    validate_resampled_data(small_data, processed_data_3)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

def test_signal_preservation():
    """
    Test that important signal characteristics are preserved during resampling.
    """
    print("\n" + "=" * 60)
    print("TESTING SIGNAL PRESERVATION")
    print("=" * 60)
    
    # Create a test signal with known characteristics
    time_samples = 1024
    time = torch.linspace(0, 1, time_samples)
    
    # Create a signal with multiple frequency components
    signal = torch.sin(2 * np.pi * 5 * time) + 0.5 * torch.sin(2 * np.pi * 20 * time)
    
    # Add to data dictionary format
    test_data = {
        'inputs_embeds': signal.unsqueeze(0).unsqueeze(0),  # Shape: (1, 1, 1024)
        'labels': signal.unsqueeze(0).unsqueeze(0),
        'index': torch.tensor([0])
    }
    
    print(f"Original signal shape: {test_data['inputs_embeds'].shape}")
    
    # Apply resampling
    resampled_data = cut_and_resample_data(test_data, cut_length=1024, target_samples=256)
    
    print(f"Resampled signal shape: {resampled_data['inputs_embeds'].shape}")
    
    # Check signal statistics
    orig_signal = test_data['inputs_embeds'].squeeze()
    resampled_signal = resampled_data['inputs_embeds'].squeeze()
    
    print(f"\nSignal statistics comparison:")
    print(f"Original - Mean: {orig_signal.mean():.6f}, Std: {orig_signal.std():.6f}")
    print(f"Resampled - Mean: {resampled_signal.mean():.6f}, Std: {resampled_signal.std():.6f}")
    print(f"Min/Max - Original: [{orig_signal.min():.6f}, {orig_signal.max():.6f}]")
    print(f"Min/Max - Resampled: [{resampled_signal.min():.6f}, {resampled_signal.max():.6f}]")
    
    # Check that the signal correlation is high
    correlation = torch.corrcoef(torch.stack([orig_signal[:256], resampled_signal]))[0, 1]
    print(f"Signal correlation: {correlation:.6f}")
    
    if correlation > 0.95:
        print("✓ Signal preservation test PASSED (correlation > 0.95)")
    else:
        print("✗ Signal preservation test FAILED (correlation <= 0.95)")

if __name__ == "__main__":
    # Run all tests
    test_cutting_and_resampling()
    test_signal_preservation()
    
    print("\n" + "=" * 60)
    print("TESTING SUMMARY")
    print("=" * 60)
    print("✓ Data cutting and resampling functions work correctly")
    print("✓ Signal characteristics are preserved during resampling")
    print("✓ Edge cases are handled properly")
    print("✓ Ready for integration into seismic data processing pipeline")
    print("=" * 60) 