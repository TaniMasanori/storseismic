# Direct Cell [16] Fix - Copy this entire code to Cell [16]
# This is a standalone solution that doesn't require external files

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
import numpy as np
from tqdm import tqdm
import time
import os

print("=== BERT Sequence Length Fix (Direct) ===")

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
    
    # Create new config with larger max_position_embeddings
    print("   Creating new BERT config...")
    new_config = BertConfig(
        vocab_size=model.config.vocab_size,
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        num_attention_heads=model.config.num_attention_heads,
        intermediate_size=model.config.intermediate_size,
        hidden_act=model.config.hidden_act,
        hidden_dropout_prob=model.config.hidden_dropout_prob,
        attention_probs_dropout_prob=model.config.attention_probs_dropout_prob,
        max_position_embeddings=actual_seq_len + 100,  # Add buffer
        type_vocab_size=2,
        initializer_range=model.config.initializer_range,
        layer_norm_eps=model.config.layer_norm_eps,
        pad_token_id=0,
        use_cache=True
    )
    
    print(f"   New max_position_embeddings: {new_config.max_position_embeddings}")
    
    # Create new model
    print("   Creating new model...")
    fixed_model = BertForMaskedLM(new_config).to(device)
    
    # Transfer weights
    print("   Transferring compatible weights...")
    old_state = model.state_dict()
    new_state = fixed_model.state_dict()
    transferred = 0
    
    for key in old_state:
        if key in new_state:
            if 'position_embeddings' in key:
                # Expand position embeddings
                old_pos = old_state[key]
                new_pos = new_state[key]
                if old_pos.shape[0] <= new_pos.shape[0]:
                    new_state[key][:old_pos.shape[0]] = old_pos
                    print(f"   ✓ Expanded position embeddings: {old_pos.shape} -> {new_pos.shape}")
            elif old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]
                transferred += 1
    
    fixed_model.load_state_dict(new_state)
    print(f"   ✓ Transferred {transferred} compatible layers")
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
    
    batch_size, seq_len = test_projected.shape[:2]
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    
    with torch.no_grad():
        test_outputs = model(inputs_embeds=test_projected, attention_mask=attention_mask)
        print(f"   ✓ Test successful: {test_inputs.shape} -> {test_projected.shape} -> {test_outputs.logits.shape}")
except Exception as e:
    print(f"   ✗ Test failed: {e}")
    raise

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
            
            # Create attention mask
            batch_size, seq_len = inputs_projected.shape[:2]
            attention_mask = torch.ones(batch_size, seq_len, device=device)
            
            # Forward pass
            outputs = model(inputs_embeds=inputs_projected, attention_mask=attention_mask)
            
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
                
                batch_size, seq_len = inputs_projected.shape[:2]
                attention_mask = torch.ones(batch_size, seq_len, device=device)
                
                outputs = model(inputs_embeds=inputs_projected, attention_mask=attention_mask)
                
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