# Fixed Training Cell - Resolves the projection layer dimension mismatch
# This cell replaces the problematic training code that caused RuntimeError

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from debug_data_shapes import debug_data_shapes, create_correct_projection
from memory_safe_data_loader import create_safe_data_loaders

def setup_training_with_correct_projection():
    """Set up training with the correct projection layer dimensions."""
    
    print("=== Setting Up Training with Correct Projection ===")
    
    # First, debug the data shapes to understand the issue
    print("Step 1: Analyzing data shapes...")
    
    try:
        # Load a small sample to analyze dimensions
        data_path = './data/pretrain/'
        train_dataloader, test_dataloader, data_loader = create_safe_data_loaders(
            data_path=data_path,
            batch_size=2,  # Small batch for analysis
            max_memory_gb=8.0
        )
        
        # Get sample batch to determine correct dimensions
        sample_batch = data_loader.get_sample_batch(train_dataloader)
        
        if sample_batch is None:
            raise RuntimeError("Could not get sample batch for analysis")
        
        # Analyze the actual data structure
        inputs_embeds = sample_batch['inputs_embeds']
        print(f"Actual inputs_embeds shape: {inputs_embeds.shape}")
        
        # Determine the correct input dimension for projection
        actual_input_dim = inputs_embeds.shape[-1]
        print(f"Detected input dimension: {actual_input_dim}")
        
        # Clean up the analysis data loader
        data_loader.cleanup()
        del train_dataloader, test_dataloader, data_loader
        torch.cuda.empty_cache()
        
        return actual_input_dim
        
    except Exception as e:
        print(f"Error during data analysis: {str(e)}")
        return None

def create_fixed_training_setup(config):
    """Create the complete training setup with correct dimensions."""
    
    print("\n=== Creating Fixed Training Setup ===")
    
    # Step 1: Determine correct projection dimensions
    actual_input_dim = setup_training_with_correct_projection()
    
    if actual_input_dim is None:
        print("Could not determine input dimensions. Using fallback approach.")
        # Try common dimensions as fallback
        possible_dims = [271, 4000, 2500, 1000]
        actual_input_dim = 271  # Default fallback
    
    print(f"Using input dimension: {actual_input_dim}")
    
    # Step 2: Load data with proper batch size
    print("\nStep 2: Loading training data...")
    
    data_path = config.dataset
    batch_size = config.batch_size
    
    # Load data with memory safety
    train_dataloader, test_dataloader, data_loader = create_safe_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        max_memory_gb=12.0
    )
    
    # Step 3: Create correct projection layer
    print(f"\nStep 3: Creating projection layer {actual_input_dim} -> 256...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    projection = nn.Linear(actual_input_dim, 256).to(device)
    
    # Step 4: Test projection with actual data
    print("\nStep 4: Testing projection layer...")
    
    try:
        sample_batch = data_loader.get_sample_batch(train_dataloader)
        test_input = sample_batch['inputs_embeds'][:1].to(device)  # Single sample test
        
        print(f"Test input shape: {test_input.shape}")
        test_output = projection(test_input.float())
        print(f"✓ Projection test successful! Output shape: {test_output.shape}")
        
    except Exception as e:
        print(f"✗ Projection test failed: {str(e)}")
        print("Attempting to fix projection dimensions...")
        
        # Try to fix by using the actual last dimension
        test_input = sample_batch['inputs_embeds'][:1]
        actual_dim = test_input.shape[-1]
        print(f"Recreating projection with dimension: {actual_dim}")
        
        projection = nn.Linear(actual_dim, 256).to(device)
        test_output = projection(test_input.to(device).float())
        print(f"✓ Fixed projection successful! Output shape: {test_output.shape}")
    
    return train_dataloader, test_dataloader, data_loader, projection, device

def run_fixed_training(model, config):
    """Run training with the fixed projection layer."""
    
    print("=== Running Fixed Training ===")
    
    # Set up training components
    train_dataloader, test_dataloader, data_loader, projection, device = create_fixed_training_setup(config)
    
    # Move model to device
    model = model.to(device)
    
    # Set up training parameters
    from radam import RAdam
    from storseismic.train import run_pretraining
    
    optim = RAdam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    epochs = config.epoch
    
    # Set up plotting
    plt.ion()
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    print(f"\nStarting training with:")
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Projection device: {next(projection.parameters()).device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Projection: {projection}")
    
    try:
        # Run training with correct projection
        model, avg_train_loss, avg_valid_loss, time_per_epoch = run_pretraining(
            model=model,
            optim=optim,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=epochs,
            device=device,
            projection=projection,  # Correctly dimensioned projection
            tmp_dir=config.parent_dir,
            patience=config.patience,
            plot=True,
            f=f,
            ax=ax
        )
        
        print("\n✓ Training completed successfully!")
        
        # Clean up
        plt.ioff()
        data_loader.cleanup()
        
        return model, avg_train_loss, avg_valid_loss, time_per_epoch
        
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        plt.ioff()
        data_loader.cleanup()
        
        return None, None, None, None

# Example usage for the notebook
if __name__ == "__main__":
    # This would be called from the notebook with the config and model
    print("Fixed training cell ready for use!")
    print("Usage in notebook:")
    print("  exec(open('fixed_training_cell.py').read())")
    print("  model, avg_train_loss, avg_valid_loss, time_per_epoch = run_fixed_training(model, config)") 