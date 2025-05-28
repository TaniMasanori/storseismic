# Debug Data Shapes - Analyze the dimension mismatch issue
# This script helps identify the actual data shapes causing the projection layer error

import torch
import os
import gc
from memory_safe_data_loader import create_safe_data_loaders

def debug_data_shapes():
    """Debug function to analyze data shapes and identify dimension mismatches."""
    
    print("=== Data Shape Debugging ===")
    
    # Load data safely
    try:
        data_path = './data/pretrain/'
        train_dataloader, test_dataloader, data_loader = create_safe_data_loaders(
            data_path=data_path,
            batch_size=2,  # Use very small batch size for debugging
            max_memory_gb=8.0
        )
        
        print("Data loaders created successfully!")
        
        # Get first batch for analysis
        sample_batch = data_loader.get_sample_batch(train_dataloader)
        
        if sample_batch is None:
            print("Error: Could not get sample batch")
            return
            
        print("\n=== Batch Analysis ===")
        print(f"Batch keys: {list(sample_batch.keys())}")
        
        for key, value in sample_batch.items():
            if hasattr(value, 'shape'):
                print(f"{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Min/Max: {value.min().item():.4f} / {value.max().item():.4f}")
                
                # Analyze specific dimensions
                if key == 'inputs_embeds':
                    print(f"  Expected shape: [batch_size, sequence_length, vocab_size]")
                    print(f"  Expected: [B, 20, 271] but got: {value.shape}")
                    
                    # Check if data needs reshaping
                    total_elements = value.numel()
                    batch_size = value.shape[0]
                    print(f"  Total elements: {total_elements}")
                    print(f"  Elements per sample: {total_elements // batch_size}")
                    
                    # Try to understand the actual structure
                    if len(value.shape) == 3:
                        print(f"  Dimension analysis:")
                        print(f"    Dim 0 (batch): {value.shape[0]}")
                        print(f"    Dim 1: {value.shape[1]}")
                        print(f"    Dim 2: {value.shape[2]}")
                        
                        # Check if it's flattened or has different structure
                        if value.shape[1] * value.shape[2] == 20 * 271:
                            print(f"  ✓ Total elements match expected 20x271 = {20*271}")
                        elif value.shape[2] == 271:
                            print(f"  ✓ Last dimension matches vocab_size (271)")
                        elif value.shape[1] == 271:
                            print(f"  ✓ Second dimension matches vocab_size (271)")
                            
                print()
        
        # Test projection layer compatibility
        print("=== Projection Layer Compatibility Test ===")
        inputs_embeds = sample_batch['inputs_embeds']
        
        print(f"Original inputs_embeds shape: {inputs_embeds.shape}")
        
        # Test different projection configurations
        test_projections = [
            (271, 256, "Standard: 271 -> 256"),
            (4000, 256, "Current error: 4000 -> 256"),
            (inputs_embeds.shape[-1], 256, f"Last dim: {inputs_embeds.shape[-1]} -> 256"),
        ]
        
        for input_dim, output_dim, description in test_projections:
            try:
                test_projection = torch.nn.Linear(input_dim, output_dim)
                print(f"\nTesting {description}")
                print(f"  Projection shape: {input_dim} -> {output_dim}")
                
                # Try to apply projection
                if input_dim == inputs_embeds.shape[-1]:
                    result = test_projection(inputs_embeds.float())
                    print(f"  ✓ Success! Output shape: {result.shape}")
                else:
                    print(f"  ✗ Dimension mismatch: input last dim is {inputs_embeds.shape[-1]}")
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        # Suggest solutions
        print("\n=== Suggested Solutions ===")
        
        actual_last_dim = inputs_embeds.shape[-1]
        if actual_last_dim != 271:
            print(f"1. Update projection layer: nn.Linear({actual_last_dim}, 256)")
            print(f"2. Or reshape data to match expected format")
            
        if len(inputs_embeds.shape) == 3 and inputs_embeds.shape[1] != 20:
            print(f"3. Check sequence length: expected 20, got {inputs_embeds.shape[1]}")
            
        # Check if data needs transposition
        if inputs_embeds.shape[1] == 271 and inputs_embeds.shape[2] != 271:
            print(f"4. Data might need transposition: current {inputs_embeds.shape}")
            print(f"   Try: inputs_embeds.transpose(1, 2)")
            
        # Clean up
        data_loader.cleanup()
        
    except Exception as e:
        print(f"Error during debugging: {str(e)}")
        import traceback
        traceback.print_exc()

def create_correct_projection(sample_batch):
    """Create the correct projection layer based on actual data dimensions."""
    
    if sample_batch is None:
        return None
        
    inputs_embeds = sample_batch['inputs_embeds']
    actual_input_dim = inputs_embeds.shape[-1]
    
    print(f"\n=== Creating Correct Projection ===")
    print(f"Actual input dimension: {actual_input_dim}")
    print(f"Target output dimension: 256")
    
    # Create projection layer with correct dimensions
    projection = torch.nn.Linear(actual_input_dim, 256)
    
    print(f"Created projection: {actual_input_dim} -> 256")
    
    # Test the projection
    try:
        test_output = projection(inputs_embeds.float())
        print(f"✓ Projection test successful! Output shape: {test_output.shape}")
        return projection
    except Exception as e:
        print(f"✗ Projection test failed: {str(e)}")
        return None

if __name__ == "__main__":
    debug_data_shapes() 