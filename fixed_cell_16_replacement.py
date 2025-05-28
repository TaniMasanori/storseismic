# Fixed Cell [16] Replacement
# Replace the content of Cell [16] with this code

# First, apply the sequence length fix
exec(open('sequence_length_quick_fix.py').read())

# Apply the fix to the model
print("Applying BERT sequence length fix...")
model, projection = apply_bert_sequence_fix(model, train_dataloader, device)

# Now run the corrected training with proper sequence length handling
print("Starting fixed training...")

# Import necessary modules for the fixed training
import time
import os
from tqdm import tqdm
import numpy as np

# Use the corrected training approach
total_time = time.time()
avg_train_loss = []
avg_valid_loss = []
time_per_epoch = []

# Early stopping setup
if hasattr(config, 'parent_dir'):
    tmp_dir = config.parent_dir
else:
    tmp_dir = './results/pretrain/'

os.makedirs(tmp_dir, exist_ok=True)
checkpoint = os.path.join(tmp_dir, str(os.getpid()) + "_fixed_checkpoint.pt")

from storseismic.pytorchtools import EarlyStopping
early_stopping = EarlyStopping(patience=config.patience if hasattr(config, 'patience') else 20, 
                              verbose=True, path=checkpoint)

epochs = 1000  # or use config.epoch if available

for epoch in range(epochs):
    epoch_time = time.time()
    model.train()
    train_losses = []

    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training loop with proper sequence handling
    for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f'Training Epoch {epoch + 1}'):
        try:
            optim.zero_grad()

            # Get data and move to device
            inputs_embeds = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)
            mask_label = batch['mask_label'].to(device)

            # Apply projection to convert from vocab_size to hidden_size
            inputs_projected = projection(inputs_embeds.float())

            # Create attention mask for the actual sequence length
            batch_size, seq_len = inputs_projected.shape[:2]
            attention_mask = torch.ones(batch_size, seq_len, device=device)

            # Forward pass with explicit attention mask
            outputs = model(inputs_embeds=inputs_projected, attention_mask=attention_mask)
            
            select_matrix = mask_label.clone()
            loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)

            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            
        except Exception as e:
            print(f"Error in training batch: {e}")
            continue

    # Validation loop
    model.eval()
    losses_valid = 0
    valid_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc=f'Validation Epoch {epoch + 1}')):
            try:
                inputs_embeds = batch['inputs_embeds'].to(device)
                mask_label = batch['mask_label'].to(device)
                labels = batch['labels'].to(device)

                inputs_projected = projection(inputs_embeds.float())
                
                batch_size, seq_len = inputs_projected.shape[:2]
                attention_mask = torch.ones(batch_size, seq_len, device=device)

                outputs = model(inputs_embeds=inputs_projected, attention_mask=attention_mask)
                
                select_matrix = mask_label.clone()
                loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)
                
                losses_valid += loss.item()
                valid_count += 1
                
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue

    if train_losses and valid_count > 0:
        avg_train_loss.append(np.mean(train_losses))
        avg_valid_loss.append(losses_valid / valid_count)
        
        print(f"Training Loss: {avg_train_loss[-1]:.4f}")
        print(f"Validation Loss: {avg_valid_loss[-1]:.4f}")
        print("Epoch time: {:.2f} s".format(time.time() - epoch_time))
        time_per_epoch.append(time.time() - epoch_time)
        print("Total time elapsed: {:.2f} s".format(time.time() - total_time))
        print("---------------------------------------")
        
        # Plotting if available
        try:
            if 'f' in globals() and 'ax' in globals():
                ax.cla()
                ax.plot(np.arange(1, epoch + 2), avg_train_loss, 'b', label='Training Loss')
                ax.plot(np.arange(1, epoch + 2), avg_valid_loss, 'orange', label='Validation Loss')
                ax.legend()
                ax.set_title("Loss Curve")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Avg Loss")
                f.canvas.draw()
        except Exception as e:
            print(f"Plotting error: {e}")

        # Early stopping
        early_stopping(avg_valid_loss[-1], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    else:
        print("No valid training data processed")
        break

# Load best model
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    print("✓ Loaded best model from checkpoint")

print("✓ Training completed successfully!")

# Save results
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ioff()

# The training variables are now available as: model, avg_train_loss, avg_valid_loss, time_per_epoch 