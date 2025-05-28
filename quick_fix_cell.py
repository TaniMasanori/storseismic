# Quick Fix Cell - One-liner solution for the projection dimension mismatch
# Copy and paste this into your notebook cell to fix the training issue

# Import required modules
import torch
import torch.nn as nn

# Quick fix: Determine correct projection dimension from loaded data
def quick_fix_projection():
    """Quickly determine and create the correct projection layer."""
    
    # Get a sample from the existing data loader to determine dimensions
    if 'train_dataloader' in globals():
        sample_batch = next(iter(train_dataloader))
        actual_input_dim = sample_batch['inputs_embeds'].shape[-1]
        
        print(f"Detected input dimension: {actual_input_dim}")
        print(f"Creating projection: {actual_input_dim} -> 256")
        
        # Create correct projection layer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        projection = nn.Linear(actual_input_dim, 256).to(device)
        
        # Test the projection
        test_input = sample_batch['inputs_embeds'][:1].to(device)
        test_output = projection(test_input.float())
        print(f"✓ Projection test successful! Input: {test_input.shape} -> Output: {test_output.shape}")
        
        return projection
    else:
        print("Error: train_dataloader not found. Please run the data loading cell first.")
        return None

# Execute the quick fix
print("=== Quick Fix for Projection Dimension Mismatch ===")
projection = quick_fix_projection()

if projection is not None:
    print("\n✓ Projection layer fixed!")
    print("You can now run your training cell with the corrected 'projection' variable.")
    print("\nExample:")
    print("model, avg_train_loss, avg_valid_loss, time_per_epoch = \\")
    print("run_pretraining(model, optim, loss_fn, train_dataloader, test_dataloader, epochs, device, projection, \\")
    print("                tmp_dir=config.parent_dir, patience=config.patience, plot=True, f=f, ax=ax)")
else:
    print("\n✗ Could not fix projection. Please check that data is loaded correctly.")

# Alternative: If you want to run the complete fixed training
print("\n" + "="*60)
print("ALTERNATIVE: Complete Fixed Training")
print("="*60)
print("If you want to run the complete fixed training process, use:")
print("exec(open('fixed_training_cell.py').read())")
print("model, avg_train_loss, avg_valid_loss, time_per_epoch = run_fixed_training(model, config)") 