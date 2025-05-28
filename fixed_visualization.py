def visualize_results_fixed(model, test_data, projection, config):
    """Fixed visualization code with proper tensor handling"""
    
    model.eval()
    
    # Debug shapes first
    inputs_embeds, labels, mask_label, sample_output = debug_tensor_shapes(
        model, test_data, projection, config
    )
    
    # Get multiple samples for visualization
    idx = torch.randint(len(test_data), (4,))
    print(f"Visualization indices: {idx}")

    inputs_embeds = test_data.encodings['inputs_embeds'][idx]
    labels = test_data.encodings['labels'][idx]
    mask_label = test_data.encodings['mask_label'][idx]

    with torch.no_grad():
        # Apply projection layer first
        inputs_projected = projection(inputs_embeds.to(device).float())
        sample_output = model(inputs_embeds=inputs_projected)

    for i, (X, y, z, mask) in enumerate(zip(inputs_embeds.cpu(), sample_output.logits.cpu(), labels.cpu(), mask_label.cpu())):
        f, ax = plt.subplots(1, 4, figsize=(15, 7.5), sharey=True, sharex=False)
        f.tight_layout()
        
        # Debug individual tensor shapes
        print(f"Sample {i}:")
        print(f"  X (input) shape: {X.shape}")
        print(f"  y (output) shape: {y.shape}")
        print(f"  z (label) shape: {z.shape}")
        print(f"  mask shape: {mask.shape}")
        
        # Fix the mean calculation based on actual tensor dimensions
        # Assuming X is [sequence_length, features] or [features, sequence_length]
        if X.dim() == 2:
            if X.shape[0] == config.max_position_embeddings:  # [max_length, vocab_size]
                X_normalized = X.detach().swapaxes(0, 1) - X.mean(dim=0, keepdim=True).swapaxes(0, 1)
                extent = [0, config.max_position_embeddings, config.vocab_size * 8 / 1000, 0]
            elif X.shape[1] == config.max_position_embeddings:  # [vocab_size, max_length]
                X_normalized = X.detach() - X.mean(dim=1, keepdim=True)
                extent = [0, config.max_position_embeddings, config.vocab_size * 8 / 1000, 0]
            else:
                # Fallback: use global mean
                X_normalized = X.detach() - X.mean()
                extent = [0, X.shape[1], X.shape[0] * 8 / 1000, 0]
        else:
            X_normalized = X.detach() - X.mean()
            extent = [0, X.shape[-1], X.shape[-2] * 8 / 1000, 0]
        
        # Similar fix for output and labels
        if y.dim() == 2:
            if y.shape[0] == config.max_position_embeddings:
                y_normalized = y.detach().swapaxes(0, 1) - y.mean(dim=0, keepdim=True).swapaxes(0, 1)
            elif y.shape[1] == config.max_position_embeddings:
                y_normalized = y.detach() - y.mean(dim=1, keepdim=True)
            else:
                y_normalized = y.detach() - y.mean()
        else:
            y_normalized = y.detach() - y.mean()
            
        if z.dim() == 2:
            if z.shape[0] == config.max_position_embeddings:
                z_normalized = z.detach().swapaxes(0, 1) - z.mean(dim=0, keepdim=True).swapaxes(0, 1)
            elif z.shape[1] == config.max_position_embeddings:
                z_normalized = z.detach() - z.mean(dim=1, keepdim=True)
            else:
                z_normalized = z.detach() - z.mean()
        else:
            z_normalized = z.detach() - z.mean()
        
        # Plot input
        ax[0].imshow(X_normalized, aspect=12, vmin=-1, vmax=1, cmap='seismic', extent=extent)
        ax[0].set_title("Input", fontsize=14)
        ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].set_xlabel("Offset Index")
        ax[0].set_yticks(np.arange(0, 2.5, 0.5))
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0].set_ylabel("t (s)")
        
        # Plot reconstructed
        ax[1].imshow(y_normalized, aspect=12, vmin=-1, vmax=1, cmap='seismic', extent=extent)
        ax[1].set_title("Reconstructed", fontsize=14)
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1].set_xlabel("Offset Index")
        
        # Plot label
        ax[2].imshow(z_normalized, aspect=12, vmin=-1, vmax=1, cmap='seismic', extent=extent)
        ax[2].set_title("Label", fontsize=14)
        ax[2].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[2].set_xlabel("Offset Index")
        
        # Plot difference
        diff = 10 * (z - y)
        im4 = ax[3].imshow(diff.detach(), aspect=12, vmin=-1, vmax=1, cmap='seismic', extent=extent)
        ax[3].set_title("10 x (Label - Reconstructed)", fontsize=14)
        ax[3].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[3].set_xlabel("Offset Index")
        
        plt.ylim(2.01, 0)
        
        # Add colorbar
        cbar_ax = f.add_axes([0.92, 0.255, 0.0125, 0.52])
        cbar = f.colorbar(im4, cax=cbar_ax)
        cbar.set_ticks(np.arange(-1, 1.25, .25))
        cbar.set_ticklabels([-1, "", "", "", "", "", "", "", 1])
        
        plt.show() 