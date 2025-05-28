import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

def visualize_seismic_results(model, test_data, projection, config, device, num_samples=4):
    """Improved visualization function for seismic data results"""
    
    model.eval()
    
    # Get random samples
    idx = torch.randint(len(test_data), (num_samples,))
    print(f"Visualization indices: {idx}")

    inputs_embeds = test_data.encodings['inputs_embeds'][idx]
    labels = test_data.encodings['labels'][idx]
    mask_label = test_data.encodings['mask_label'][idx]

    with torch.no_grad():
        # Apply projection layer first
        inputs_projected = projection(inputs_embeds.to(device).float())
        sample_output = model(inputs_embeds=inputs_projected)

    # Create plots for each sample
    for i, (X, y, z, mask) in enumerate(zip(inputs_embeds.cpu(), sample_output.logits.cpu(), labels.cpu(), mask_label.cpu())):
        # Create figure
        fig, ax = plt.subplots(1, 4, figsize=(15, 7.5), sharey=True, sharex=False)
        fig.suptitle(f'Sample {i+1}', fontsize=16, fontweight='bold')
        fig.tight_layout()
        
        # Print tensor shapes for debugging
        print(f"Sample {i+1} shapes - X: {X.shape}, y: {y.shape}, z: {z.shape}")
        
        # Handle different tensor dimensions
        if X.dim() == 2:
            # Assume [vocab_size, max_length] format
            extent = [0, config.max_position_embeddings, config.vocab_size * 8 / 1000, 0]
            
            # Input plot
            X_normalized = X.detach().swapaxes(0, 1) - X.mean()
            im1 = ax[0].imshow(X_normalized, aspect=12, vmin=-1, vmax=1, cmap='seismic', extent=extent)
            ax[0].set_title("Input", fontsize=14)
            ax[0].set_xlabel("Offset Index")
            ax[0].set_ylabel("t (s)")
            ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[0].set_yticks(np.arange(0, 2.5, 0.5))
            ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
            
            # Reconstructed plot
            y_normalized = y.detach().swapaxes(0, 1) - y.mean()
            im2 = ax[1].imshow(y_normalized, aspect=12, vmin=-1, vmax=1, cmap='seismic', extent=extent)
            ax[1].set_title("Reconstructed", fontsize=14)
            ax[1].set_xlabel("Offset Index")
            ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
            
            # Label plot
            z_normalized = z.detach().swapaxes(0, 1) - z.mean()
            im3 = ax[2].imshow(z_normalized, aspect=12, vmin=-1, vmax=1, cmap='seismic', extent=extent)
            ax[2].set_title("Label", fontsize=14)
            ax[2].set_xlabel("Offset Index")
            ax[2].xaxis.set_minor_locator(AutoMinorLocator(5))
            
            # Difference plot
            diff = 10 * (z - y)
            diff_normalized = diff.detach().swapaxes(0, 1)
            im4 = ax[3].imshow(diff_normalized, aspect=12, vmin=-1, vmax=1, cmap='seismic', extent=extent)
            ax[3].set_title("10 × (Label - Reconstructed)", fontsize=14)
            ax[3].set_xlabel("Offset Index")
            ax[3].xaxis.set_minor_locator(AutoMinorLocator(5))
            
        else:
            print(f"Unexpected tensor dimension: {X.dim()}")
            continue
        
        # Set y-axis limits
        for a in ax:
            a.set_ylim(2.01, 0)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.255, 0.0125, 0.52])
        cbar = fig.colorbar(im4, cax=cbar_ax)
        cbar.set_ticks(np.arange(-1, 1.25, .25))
        cbar.set_ticklabels([-1, "", "", "", "", "", "", "", 1])
        
        # Show the plot
        plt.show()
        
        # Optional: Save the plot
        # plt.savefig(f'seismic_reconstruction_sample_{i+1}.png', dpi=300, bbox_inches='tight')
    
    print(f"Displayed {num_samples} visualization plots")

# Usage example:
def run_visualization(model, test_data, projection, config, device):
    """Run the visualization with error handling"""
    try:
        visualize_seismic_results(model, test_data, projection, config, device, num_samples=4)
    except Exception as e:
        print(f"Visualization error: {e}")
        # Fallback to simple debug
        debug_tensor_shapes(model, test_data, projection, config, device)

def debug_tensor_shapes(model, test_data, projection, config, device):
    """Debug function to check tensor shapes"""
    model.eval()
    
    idx = torch.randint(len(test_data), (1,))
    print(f"Debug - Sample index: {idx}")
    
    inputs_embeds = test_data.encodings['inputs_embeds'][idx]
    labels = test_data.encodings['labels'][idx]
    
    print(f"Original inputs_embeds shape: {inputs_embeds.shape}")
    print(f"Original labels shape: {labels.shape}")
    
    with torch.no_grad():
        inputs_projected = projection(inputs_embeds.to(device).float())
        print(f"Projected inputs shape: {inputs_projected.shape}")
        
        sample_output = model(inputs_embeds=inputs_projected)
        print(f"Model output logits shape: {sample_output.logits.shape}") 