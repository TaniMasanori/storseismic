# Safe Data Visualization Cell - Replacement for the visualization cell
# This cell safely visualizes the loaded data without causing memory issues

import matplotlib.pyplot as plt
import torch
import gc

print("=== Safe Data Visualization ===")

# Check if data loaders exist
if 'train_dataloader' not in locals() or 'test_dataloader' not in locals():
    print("Error: Data loaders not found. Please run the safe data loading cell first.")
else:
    try:
        # Get batch size from config or dataloader
        batch_size = getattr(train_dataloader, 'batch_size', 256)
        
        print(f"Visualizing data with batch size: {batch_size}")
        print(f"Training batches available: {len(train_dataloader)}")
        print(f"Test batches available: {len(test_dataloader)}")
        
        # Safely iterate through first batch
        for i, X in enumerate(train_dataloader):
            if i == 0:  # Only process first batch
                print(f"Batch shape information:")
                for key, value in X.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                
                # Visualize first 4 samples (or fewer if batch is smaller)
                num_samples = min(4, X['inputs_embeds'].shape[0])
                
                for j in range(num_samples):
                    try:
                        f, ax = plt.subplots(1, 2, figsize=(10, 5))
                        
                        # Input visualization
                        input_data = X['inputs_embeds'][j, :, :].swapaxes(0, 1) - X['inputs_embeds'][j, :, :].mean()
                        ax[0].imshow(input_data, aspect=.1, vmin=vmin_all, vmax=vmax_all, cmap='seismic')
                        ax[0].set_title(f"Input - Sample {j+1}")
                        
                        # Label visualization
                        label_data = X['labels'][j, :, :].swapaxes(0, 1) - X['labels'][j, :, :].mean()
                        ax[1].imshow(label_data, aspect=.1, vmin=vmin_all, vmax=vmax_all, cmap='seismic')
                        ax[1].set_title(f"Label - Sample {j+1}")
                        
                        plt.tight_layout()
                        plt.show()
                        
                        # Clear memory after each plot
                        del input_data, label_data
                        
                    except Exception as e:
                        print(f"Error visualizing sample {j+1}: {str(e)}")
                        continue
                
                # Clean up batch data
                del X
                break
            
        # Force garbage collection
        gc.collect()
        
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        print("This might be due to memory constraints or data format issues.")
        
        # Try to get basic info about the data
        try:
            sample_batch = next(iter(train_dataloader))
            print(f"Available keys in data: {list(sample_batch.keys())}")
            del sample_batch
        except Exception as e2:
            print(f"Could not access data: {str(e2)}")
        
        # Clean up
        gc.collect()

print("=== Visualization Complete ===")
print("Memory cleanup performed. Ready for next steps.") 