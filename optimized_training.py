import torch
from torch.cuda.amp import GradScaler, autocast
from memory_optimization import setup_memory_optimization, clear_memory, monitor_memory

def optimized_training_loop(model, train_dataloader, test_dataloader, config, projection, optim, loss_fn, device):
    """Memory-optimized training loop with gradient accumulation and mixed precision"""
    
    # Setup memory optimization
    setup_memory_optimization()
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp else None
    
    # Training tracking
    avg_train_loss = []
    avg_valid_loss = []
    
    monitor_memory("Before training")
    
    for epoch in range(config.epoch):
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        # Clear memory at start of epoch
        clear_memory()
        
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                # Move data to device
                inputs_embeds = batch['inputs_embeds'].to(device, non_blocking=True).float()
                labels = batch['labels'].to(device, non_blocking=True).float()
                
                # Forward pass with mixed precision
                with autocast(enabled=config.use_amp):
                    # Apply projection
                    inputs_projected = projection(inputs_embeds)
                    
                    # Model forward
                    outputs = model(inputs_embeds=inputs_projected)
                    
                    # Calculate loss
                    loss = loss_fn(outputs.logits, labels)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                train_loss += loss.item()
                
                # Gradient accumulation
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    
                    optim.zero_grad()
                    
                    # Clear cache periodically
                    if (batch_idx + 1) % (config.gradient_accumulation_steps * 4) == 0:
                        clear_memory()
                
                num_train_batches += 1
                
                # Monitor memory every 10 batches
                if batch_idx % 10 == 0:
                    monitor_memory(f"Epoch {epoch}, Batch {batch_idx}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in training batch {batch_idx}: {e}")
                    clear_memory()
                    continue
                else:
                    raise e
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                try:
                    inputs_embeds = batch['inputs_embeds'].to(device, non_blocking=True).float()
                    labels = batch['labels'].to(device, non_blocking=True).float()
                    
                    with autocast(enabled=config.use_amp):
                        inputs_projected = projection(inputs_embeds)
                        outputs = model(inputs_embeds=inputs_projected)
                        loss = loss_fn(outputs.logits, labels)
                    
                    valid_loss += loss.item()
                    num_valid_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in validation batch {batch_idx}: {e}")
                        clear_memory()
                        continue
                    else:
                        raise e
        
        # Calculate averages
        if num_train_batches > 0:
            avg_train_loss.append(train_loss / num_train_batches)
        if num_valid_batches > 0:
            avg_valid_loss.append(valid_loss / num_valid_batches)
        
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss[-1]:.4f}, Valid Loss: {avg_valid_loss[-1]:.4f}")
        
        # Clear memory at end of epoch
        clear_memory()
    
    return avg_train_loss, avg_valid_loss 