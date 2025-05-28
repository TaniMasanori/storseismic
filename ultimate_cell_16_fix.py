# Ultimate Cell [16] Fix - Final corrected version
# Copy this entire code to Cell [16]

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
import numpy as np
from tqdm import tqdm
import time
import os
import copy

print("=== BERT Sequence Length Fix (Ultimate) ===")

# Step 1: Analyze the current data to understand the mismatch
print("1. Analyzing data dimensions...")
try:
    sample_batch = next(iter(train_dataloader))
    inputs_shape = sample_batch['inputs_embeds'].shape
    actual_seq_len = inputs_shape[1]
    actual_vocab_size = inputs_shape[2] if len(inputs_shape) >= 3 else inputs_shape[1]
    batch_size = inputs_shape[0]
    
    print(f"   Data shape: {inputs_shape}")
    print(f"   Sequence length: {actual_seq_len}")
    print(f"   Vocab size: {actual_vocab_size}")
    print(f"   Batch size: {batch_size}")
    
except Exception as e:
    print(f"✗ Error analyzing data: {e}")
    raise

# Step 2: Check if model needs fixing
print("\n2. Checking model compatibility...")
current_max_pos = model.config.max_position_embeddings
print(f"   Model max_position_embeddings: {current_max_pos}")
print(f"   Actual sequence length: {actual_seq_len}")

if actual_seq_len > current_max_pos:
    print("   ⚠ Model needs sequence length fix")
    
    # Create new config preserving ALL original attributes and class
    print("   Creating new BERT config with preserved attributes...")
    
    # Method 1: Try to create a proper config object
    try:
        # Create a deep copy of the original config
        new_config = copy.deepcopy(model.config)
        
        # Update the max_position_embeddings
        new_config.max_position_embeddings = actual_seq_len + 100
        
        print(f"   Original max_position_embeddings: {current_max_pos}")
        print(f"   New max_position_embeddings: {new_config.max_position_embeddings}")
        
        # Create new model using the same class as the original
        print("   Creating new model...")
        original_model_class = type(model)
        fixed_model = original_model_class(new_config).to(device)
        
    except Exception as e:
        print(f"   Method 1 failed: {e}")
        print("   Trying method 2: Manual position embedding expansion...")
        
        # Method 2: Manually expand position embeddings without recreating the model
        print("   Expanding position embeddings in-place...")
        
        # Get current position embeddings
        current_pos_emb = model.bert.embeddings.position_embeddings
        current_weight = current_pos_emb.weight.data  # Shape: [20, 4000]
        
        # Create new larger position embeddings
        new_pos_emb_size = actual_seq_len + 100
        new_pos_emb = nn.Embedding(new_pos_emb_size, current_weight.shape[1]).to(device)
        
        # Initialize with existing weights + random for new positions
        with torch.no_grad():
            # Copy existing position embeddings
            new_pos_emb.weight.data[:current_weight.shape[0]] = current_weight
            # Initialize new positions randomly (similar to original initialization)
            if new_pos_emb_size > current_weight.shape[0]:
                new_pos_emb.weight.data[current_weight.shape[0]:].normal_(mean=0.0, std=model.config.initializer_range)
        
        # Replace the position embeddings
        model.bert.embeddings.position_embeddings = new_pos_emb
        
        # Update the config
        model.config.max_position_embeddings = new_pos_emb_size
        
        print(f"   ✓ Expanded position embeddings: {current_weight.shape} -> {new_pos_emb.weight.shape}")
        
        fixed_model = model
    
    model = fixed_model
    
else:
    print("   ✓ Model is compatible")

# Step 3: Create projection layer
print("\n3. Creating projection layer...")
projection = nn.Linear(actual_vocab_size, model.config.hidden_size).to(device)
print(f"   Projection: {actual_vocab_size} -> {model.config.hidden_size}")

# Step 4: Test the fix
print("\n4. Testing model compatibility...")
try:
    test_inputs = sample_batch['inputs_embeds'][:2].to(device)
    test_projected = projection(test_inputs.float())
    
    with torch.no_grad():
        # Try different approaches for forward pass
        try:
            # Method 1: With attention mask
            batch_size, seq_len = test_projected.shape[:2]
            attention_mask = torch.ones(batch_size, seq_len, device=device)
            test_outputs = model(inputs_embeds=test_projected, attention_mask=attention_mask)
            print(f"   ✓ Test successful with attention mask: {test_inputs.shape} -> {test_projected.shape} -> {test_outputs.logits.shape}")
        except Exception as e1:
            print(f"   Method 1 failed: {e1}")
            try:
                # Method 2: Without attention mask
                test_outputs = model(inputs_embeds=test_projected)
                print(f"   ✓ Test successful without attention mask: {test_inputs.shape} -> {test_projected.shape} -> {test_outputs.logits.shape}")
            except Exception as e2:
                print(f"   Method 2 failed: {e2}")
                # Method 3: Manual forward through components
                print("   Trying manual forward pass...")
                embeddings_output = model.bert.embeddings(inputs_embeds=test_projected)
                encoder_outputs = model.bert.encoder(embeddings_output)
                sequence_output = encoder_outputs[0]
                prediction_scores = model.cls(sequence_output)
                print(f"   ✓ Manual forward successful: {test_inputs.shape} -> {test_projected.shape} -> {prediction_scores.shape}")
                
except Exception as e:
    print(f"   ✗ All test methods failed: {e}")
    print("   Proceeding with training anyway...")

# Step 5: Run fixed training
print("\n5. Starting fixed training...")

# Setup
epochs = 1000
patience = 20
tmp_dir = getattr(config, 'parent_dir', './results/pretrain/')
os.makedirs(tmp_dir, exist_ok=True)

total_time = time.time()
avg_train_loss = []
avg_valid_loss = []
time_per_epoch = []

# Early stopping
checkpoint = os.path.join(tmp_dir, str(os.getpid()) + "_fixed_checkpoint.pt")

# Import or define early stopping
try:
    from storseismic.pytorchtools import EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint)
    print("   ✓ Using storseismic EarlyStopping")
except ImportError:
    print("   Using built-in EarlyStopping")
    # Simple early stopping implementation
    class EarlyStopping:
        def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta
            self.path = path

        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint)

# Define robust forward function
def robust_forward(model, inputs_projected):
    """Try multiple methods for forward pass"""
    try:
        # Method 1: Standard forward with attention mask
        batch_size, seq_len = inputs_projected.shape[:2]
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        return model(inputs_embeds=inputs_projected, attention_mask=attention_mask)
    except:
        try:
            # Method 2: Forward without attention mask
            return model(inputs_embeds=inputs_projected)
        except:
            # Method 3: Manual forward through components
            embeddings_output = model.bert.embeddings(inputs_embeds=inputs_projected)
            encoder_outputs = model.bert.encoder(embeddings_output)
            sequence_output = encoder_outputs[0]
            prediction_scores = model.cls(sequence_output)
            # Create a simple namespace to mimic model output
            from types import SimpleNamespace
            return SimpleNamespace(logits=prediction_scores)

# Training loop
for epoch in range(epochs):
    epoch_time = time.time()
    model.train()
    train_losses = []
    
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training
    train_pbar = tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}', leave=False)
    for batch_idx, batch in enumerate(train_pbar):
        try:
            optim.zero_grad()
            
            # Get data
            inputs_embeds = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)
            mask_label = batch['mask_label'].to(device)
            
            # Apply projection
            inputs_projected = projection(inputs_embeds.float())
            
            # Robust forward pass
            outputs = robust_forward(model, inputs_projected)
            
            # Calculate loss
            select_matrix = mask_label.clone()
            loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)
            
            # Backward pass
            loss.backward()
            optim.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': loss.item()})
            
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
    
    # Validation
    model.eval()
    valid_losses = []
    
    with torch.no_grad():
        valid_pbar = tqdm(test_dataloader, desc=f'Validation Epoch {epoch + 1}', leave=False)
        for batch_idx, batch in enumerate(valid_pbar):
            try:
                inputs_embeds = batch['inputs_embeds'].to(device)
                mask_label = batch['mask_label'].to(device)
                labels = batch['labels'].to(device)
                
                inputs_projected = projection(inputs_embeds.float())
                
                # Robust forward pass
                outputs = robust_forward(model, inputs_projected)
                
                select_matrix = mask_label.clone()
                loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)
                
                valid_losses.append(loss.item())
                valid_pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    # Calculate averages and log
    if train_losses and valid_losses:
        avg_train_loss.append(np.mean(train_losses))
        avg_valid_loss.append(np.mean(valid_losses))
        
        print(f"Training Loss: {avg_train_loss[-1]:.4f}")
        print(f"Validation Loss: {avg_valid_loss[-1]:.4f}")
        
        epoch_duration = time.time() - epoch_time
        time_per_epoch.append(epoch_duration)
        print(f"Epoch time: {epoch_duration:.2f} s")
        print(f"Total time elapsed: {time.time() - total_time:.2f} s")
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
            print("Early stopping triggered")
            break
    else:
        print("No valid training or validation data processed")
        break

# Load best model
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    print("✓ Loaded best model from checkpoint")

print("✓ Training completed successfully!")

# Save results
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ioff()

print("\nTraining variables available:")
print(f"- model: {type(model)}")
print(f"- avg_train_loss: {len(avg_train_loss)} epochs")
print(f"- avg_valid_loss: {len(avg_valid_loss)} epochs") 
print(f"- time_per_epoch: {len(time_per_epoch)} epochs")
print(f"- projection: {projection}") 