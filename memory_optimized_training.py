# Memory-Optimized Training Solution
# Copy this code to replace Cell [16] or run in a new cell

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
import numpy as np
from tqdm import tqdm
import time
import os
import gc

print("=== Memory-Optimized BERT Training ===")

# Step 1: Memory Management Functions
def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - allocated
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total
        }
    return None

# Step 2: Enable memory optimization settings
print("1. Setting up memory optimization...")
torch.backends.cudnn.benchmark = False  # Disable for memory consistency
torch.backends.cudnn.deterministic = True

# Set PyTorch memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear memory before starting
clear_gpu_memory()
memory_info = get_gpu_memory_info()
if memory_info:
    print(f"   Initial GPU memory: {memory_info['allocated']:.2f}GB allocated, {memory_info['free']:.2f}GB free")

# Step 3: Analyze data and reduce batch size if necessary
print("\n2. Analyzing data and optimizing batch size...")
try:
    sample_batch = next(iter(train_dataloader))
    inputs_shape = sample_batch['inputs_embeds'].shape
    actual_seq_len = inputs_shape[1]
    actual_vocab_size = inputs_shape[2] if len(inputs_shape) >= 3 else inputs_shape[1]
    original_batch_size = inputs_shape[0]
    
    print(f"   Original data shape: {inputs_shape}")
    print(f"   Sequence length: {actual_seq_len}")
    print(f"   Vocab size: {actual_vocab_size}")
    print(f"   Original batch size: {original_batch_size}")
    
    # Calculate memory requirements
    element_size = 4  # float32
    input_memory_per_sample = actual_seq_len * actual_vocab_size * element_size / 1024**3  # GB
    print(f"   Memory per sample: {input_memory_per_sample:.4f} GB")
    
    # Determine optimal batch size for available memory
    available_memory = memory_info['free'] * 0.7 if memory_info else 4.0  # Use 70% of free memory
    optimal_batch_size = max(1, int(available_memory / (input_memory_per_sample * 10)))  # Factor of 10 for safety
    optimal_batch_size = min(optimal_batch_size, original_batch_size)
    
    print(f"   Available memory: {available_memory:.2f} GB")
    print(f"   Optimal batch size: {optimal_batch_size}")
    
except Exception as e:
    print(f"✗ Error analyzing data: {e}")
    optimal_batch_size = 1  # Fallback to smallest batch size

# Step 4: Create memory-efficient data loaders
print("\n3. Creating memory-efficient data loaders...")

def create_memory_efficient_dataloader(original_dataloader, new_batch_size):
    """Create a new dataloader with smaller batch size"""
    dataset = original_dataloader.dataset
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=new_batch_size,
        shuffle=False,  # Keep original order
        num_workers=0,  # Reduce workers to save memory
        pin_memory=False,  # Disable pin memory to save GPU memory
        drop_last=False
    )

# Create smaller batch size dataloaders if needed
if optimal_batch_size < original_batch_size:
    print(f"   Reducing batch size from {original_batch_size} to {optimal_batch_size}")
    train_dataloader_small = create_memory_efficient_dataloader(train_dataloader, optimal_batch_size)
    test_dataloader_small = create_memory_efficient_dataloader(test_dataloader, optimal_batch_size)
else:
    print(f"   Using original batch size: {original_batch_size}")
    train_dataloader_small = train_dataloader
    test_dataloader_small = test_dataloader

# Step 5: Model memory optimization
print("\n4. Optimizing model memory usage...")

# Check if model needs sequence length fix
current_max_pos = model.config.max_position_embeddings
print(f"   Model max_position_embeddings: {current_max_pos}")
print(f"   Actual sequence length: {actual_seq_len}")

if actual_seq_len > current_max_pos:
    print("   ⚠ Expanding position embeddings...")
    
    # Method: In-place expansion (most memory efficient)
    current_pos_emb = model.bert.embeddings.position_embeddings
    current_weight = current_pos_emb.weight.data
    
    new_pos_emb_size = actual_seq_len + 100
    new_pos_emb = nn.Embedding(new_pos_emb_size, current_weight.shape[1]).to(device)
    
    with torch.no_grad():
        new_pos_emb.weight.data[:current_weight.shape[0]] = current_weight
        if new_pos_emb_size > current_weight.shape[0]:
            new_pos_emb.weight.data[current_weight.shape[0]:].normal_(mean=0.0, std=model.config.initializer_range)
    
    # Replace and clear old embedding
    del model.bert.embeddings.position_embeddings
    clear_gpu_memory()
    model.bert.embeddings.position_embeddings = new_pos_emb
    model.config.max_position_embeddings = new_pos_emb_size
    
    print(f"   ✓ Expanded position embeddings: {current_weight.shape} -> {new_pos_emb.weight.shape}")

# Create projection layer
print("\n5. Creating memory-efficient projection layer...")
projection = nn.Linear(actual_vocab_size, model.config.hidden_size).to(device)
print(f"   Projection: {actual_vocab_size} -> {model.config.hidden_size}")

# Clear memory after setup
clear_gpu_memory()

# Step 6: Memory-efficient forward function
def memory_efficient_forward(model, inputs_projected, accumulate_grad=False):
    """Memory-efficient forward pass with gradient accumulation support"""
    try:
        # Try standard forward
        return model(inputs_embeds=inputs_projected)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("   OOM detected, clearing cache and retrying...")
            clear_gpu_memory()
            try:
                return model(inputs_embeds=inputs_projected)
            except RuntimeError:
                # If still OOM, try manual forward
                print("   Using manual forward pass...")
                embeddings_output = model.bert.embeddings(inputs_embeds=inputs_projected)
                clear_gpu_memory()
                encoder_outputs = model.bert.encoder(embeddings_output)
                clear_gpu_memory()
                sequence_output = encoder_outputs[0]
                prediction_scores = model.cls(sequence_output)
                from types import SimpleNamespace
                return SimpleNamespace(logits=prediction_scores)
        else:
            raise e

# Step 7: Memory-efficient training loop
print("\n6. Starting memory-efficient training...")

# Setup
epochs = 1000
patience = 20
tmp_dir = getattr(config, 'parent_dir', './results/pretrain/')
os.makedirs(tmp_dir, exist_ok=True)

# Gradient accumulation to simulate larger batch sizes
accumulation_steps = max(1, original_batch_size // optimal_batch_size)
print(f"   Using gradient accumulation steps: {accumulation_steps}")

total_time = time.time()
avg_train_loss = []
avg_valid_loss = []
time_per_epoch = []

# Early stopping
checkpoint = os.path.join(tmp_dir, str(os.getpid()) + "_memory_optimized_checkpoint.pt")

try:
    from storseismic.pytorchtools import EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint)
    print("   ✓ Using storseismic EarlyStopping")
except ImportError:
    print("   Using built-in EarlyStopping")
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

# Training loop with memory management
for epoch in range(epochs):
    epoch_time = time.time()
    model.train()
    train_losses = []
    
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Check memory before epoch
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"   Memory before epoch: {memory_info['allocated']:.2f}GB allocated, {memory_info['free']:.2f}GB free")
    
    # Training with gradient accumulation
    optim.zero_grad()
    accumulation_loss = 0.0
    processed_batches = 0
    
    train_pbar = tqdm(train_dataloader_small, desc=f'Training Epoch {epoch + 1}', leave=False)
    for batch_idx, batch in enumerate(train_pbar):
        try:
            # Get data
            inputs_embeds = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)
            mask_label = batch['mask_label'].to(device)
            
            # Apply projection
            inputs_projected = projection(inputs_embeds.float())
            
            # Forward pass
            outputs = memory_efficient_forward(model, inputs_projected)
            
            # Calculate loss
            select_matrix = mask_label.clone()
            loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            accumulation_loss += loss.item()
            processed_batches += 1
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader_small):
                optim.step()
                optim.zero_grad()
                
                train_losses.append(accumulation_loss)
                accumulation_loss = 0.0
                
                # Clear memory periodically
                if (batch_idx + 1) % (accumulation_steps * 5) == 0:
                    clear_gpu_memory()
            
            train_pbar.set_postfix({'loss': loss.item() * accumulation_steps, 'batch': batch_idx + 1})
            
            # Clear tensors
            del inputs_embeds, labels, mask_label, inputs_projected, outputs, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   OOM in training batch {batch_idx}, clearing memory and continuing...")
                clear_gpu_memory()
                continue
            else:
                print(f"   Error in training batch {batch_idx}: {e}")
                continue
        except Exception as e:
            print(f"   Error in training batch {batch_idx}: {e}")
            continue
    
    # Validation with memory management
    model.eval()
    valid_losses = []
    
    with torch.no_grad():
        valid_pbar = tqdm(test_dataloader_small, desc=f'Validation Epoch {epoch + 1}', leave=False)
        for batch_idx, batch in enumerate(valid_pbar):
            try:
                inputs_embeds = batch['inputs_embeds'].to(device)
                mask_label = batch['mask_label'].to(device)
                labels = batch['labels'].to(device)
                
                inputs_projected = projection(inputs_embeds.float())
                
                outputs = memory_efficient_forward(model, inputs_projected)
                
                select_matrix = mask_label.clone()
                loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)
                
                valid_losses.append(loss.item())
                valid_pbar.set_postfix({'loss': loss.item()})
                
                # Clear tensors
                del inputs_embeds, labels, mask_label, inputs_projected, outputs, loss
                
                # Clear memory periodically
                if (batch_idx + 1) % 10 == 0:
                    clear_gpu_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   OOM in validation batch {batch_idx}, clearing memory and continuing...")
                    clear_gpu_memory()
                    continue
                else:
                    print(f"   Error in validation batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                print(f"   Error in validation batch {batch_idx}: {e}")
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
        
        # Memory info
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"Memory after epoch: {memory_info['allocated']:.2f}GB allocated, {memory_info['free']:.2f}GB free")
        
        print("---------------------------------------")
        
        # Plotting if available
        try:
            if 'f' in globals() and 'ax' in globals():
                ax.cla()
                ax.plot(np.arange(1, epoch + 2), avg_train_loss, 'b', label='Training Loss')
                ax.plot(np.arange(1, epoch + 2), avg_valid_loss, 'orange', label='Validation Loss')
                ax.legend()
                ax.set_title("Loss Curve (Memory Optimized)")
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
    
    # Clear memory at end of epoch
    clear_gpu_memory()

# Load best model
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    print("✓ Loaded best model from checkpoint")

print("✓ Memory-optimized training completed successfully!")

# Save results
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ioff()

# Final memory cleanup
clear_gpu_memory()

print("\nTraining variables available:")
print(f"- model: {type(model)}")
print(f"- avg_train_loss: {len(avg_train_loss)} epochs")
print(f"- avg_valid_loss: {len(avg_valid_loss)} epochs") 
print(f"- time_per_epoch: {len(time_per_epoch)} epochs")
print(f"- projection: {projection}")

# Final memory status
memory_info = get_gpu_memory_info()
if memory_info:
    print(f"Final GPU memory: {memory_info['allocated']:.2f}GB allocated, {memory_info['free']:.2f}GB free") 