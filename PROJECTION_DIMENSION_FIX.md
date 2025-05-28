# Projection Dimension Mismatch Fix

## Problem Description

The training process failed with a `RuntimeError` indicating that matrix multiplication shapes are incompatible:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (255744x4000 and 271x256)
```

This error occurs in the projection layer where the input data dimensions don't match the expected projection layer input dimensions.

## Root Cause Analysis

### Expected vs Actual Dimensions

- **Expected input shape**: `[batch_size, sequence_length, vocab_size]` = `[B, 20, 271]`
- **Expected projection**: `nn.Linear(271, 256)` (vocab_size → hidden_size)
- **Actual input shape**: `[batch_size, flattened_dimension]` where flattened_dimension ≠ 271

### Why This Happens

1. **Data preprocessing differences**: The data might have been preprocessed differently than expected
2. **Flattened data**: The data might be flattened instead of maintaining the expected 3D structure
3. **Different vocab_size**: The actual vocabulary size might be different from the configured 271
4. **Batch reshaping**: The data might be reshaped during loading, changing the expected dimensions

## Solutions

### Solution 1: Quick Fix (Recommended for immediate resolution)

**Use this if you want to fix the issue immediately:**

```python
# Copy and paste this into a new notebook cell
exec(open('quick_fix_cell.py').read())
```

This will:
1. Automatically detect the correct input dimensions from your loaded data
2. Create a properly sized projection layer
3. Test the projection to ensure it works
4. Provide the corrected `projection` variable for your training

### Solution 2: Complete Fixed Training

**Use this for a comprehensive solution:**

```python
# Copy and paste this into a new notebook cell
exec(open('fixed_training_cell.py').read())
model, avg_train_loss, avg_valid_loss, time_per_epoch = run_fixed_training(model, config)
```

This will:
1. Analyze your data dimensions
2. Create the correct projection layer
3. Set up memory-safe data loading
4. Run the complete training process with proper error handling

### Solution 3: Manual Fix

**If you prefer to understand and fix manually:**

1. **Check your data dimensions:**
   ```python
   sample_batch = next(iter(train_dataloader))
   print(f"inputs_embeds shape: {sample_batch['inputs_embeds'].shape}")
   ```

2. **Create correct projection:**
   ```python
   actual_input_dim = sample_batch['inputs_embeds'].shape[-1]
   projection = nn.Linear(actual_input_dim, 256).to(device)
   ```

3. **Test the projection:**
   ```python
   test_output = projection(sample_batch['inputs_embeds'][:1].to(device).float())
   print(f"Projection test: {sample_batch['inputs_embeds'][:1].shape} -> {test_output.shape}")
   ```

### Solution 4: Debug and Analyze

**For detailed analysis of the issue:**

```python
exec(open('debug_data_shapes.py').read())
debug_data_shapes()
```

This will provide comprehensive analysis of:
- Actual data shapes and dimensions
- Expected vs actual formats
- Suggested fixes
- Compatibility tests

## Common Scenarios and Fixes

### Scenario 1: Data is flattened
**Problem**: Data shape is `[batch_size, flattened_features]` instead of `[batch_size, seq_len, vocab_size]`

**Fix**: 
```python
# If data is flattened, reshape it
if len(inputs_embeds.shape) == 2:
    inputs_embeds = inputs_embeds.view(batch_size, 20, -1)  # Reshape to [B, 20, vocab_size]
```

### Scenario 2: Different vocab_size
**Problem**: Actual vocab_size is different from expected 271

**Fix**:
```python
# Use actual last dimension for projection
actual_vocab_size = inputs_embeds.shape[-1]
projection = nn.Linear(actual_vocab_size, 256).to(device)
```

### Scenario 3: Transposed dimensions
**Problem**: Data has shape `[batch_size, vocab_size, seq_len]` instead of `[batch_size, seq_len, vocab_size]`

**Fix**:
```python
# Transpose the last two dimensions
inputs_embeds = inputs_embeds.transpose(-2, -1)
```

### Scenario 4: Wrong batch processing
**Problem**: Data is processed incorrectly during batching

**Fix**: Use the memory-safe data loader which handles batching correctly:
```python
from memory_safe_data_loader import create_safe_data_loaders
train_dataloader, test_dataloader, data_loader = create_safe_data_loaders(
    data_path='./data/pretrain/',
    batch_size=256,
    max_memory_gb=12.0
)
```

## Prevention

### For Future Data Processing

1. **Validate data shapes** after loading:
   ```python
   assert inputs_embeds.shape[-1] == config.vocab_size, f"Expected vocab_size {config.vocab_size}, got {inputs_embeds.shape[-1]}"
   ```

2. **Use consistent data preprocessing**:
   ```python
   # Ensure data maintains expected format [B, seq_len, vocab_size]
   if len(data.shape) != 3:
       raise ValueError(f"Expected 3D data [B, seq_len, vocab_size], got {data.shape}")
   ```

3. **Document data format** in your preprocessing scripts

### For Model Configuration

1. **Match config to actual data**:
   ```python
   # Update config based on actual data
   sample_batch = next(iter(train_dataloader))
   config.vocab_size = sample_batch['inputs_embeds'].shape[-1]
   config.max_position_embeddings = sample_batch['inputs_embeds'].shape[-2]
   ```

2. **Use dynamic projection creation**:
   ```python
   # Create projection based on actual data dimensions
   def create_adaptive_projection(sample_data, hidden_size=256):
       input_dim = sample_data.shape[-1]
       return nn.Linear(input_dim, hidden_size)
   ```

## Troubleshooting

### If Quick Fix Doesn't Work

1. **Check data loading**:
   ```python
   # Verify data is loaded correctly
   print(f"train_dataloader exists: {'train_dataloader' in globals()}")
   print(f"Data loader length: {len(train_dataloader) if 'train_dataloader' in globals() else 'N/A'}")
   ```

2. **Check memory issues**:
   ```python
   # Use smaller batch size for testing
   test_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
   ```

3. **Check device compatibility**:
   ```python
   # Ensure all tensors are on the same device
   print(f"Model device: {next(model.parameters()).device}")
   print(f"Data device: {sample_batch['inputs_embeds'].device}")
   ```

### Common Error Messages

- **"mat1 and mat2 shapes cannot be multiplied"**: Dimension mismatch in projection layer
- **"Expected 3D tensor, got 2D"**: Data reshaping issue
- **"CUDA out of memory"**: Use memory-safe data loading
- **"RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same"**: Device mismatch

## Files Created for This Fix

1. **`quick_fix_cell.py`**: Immediate one-liner fix
2. **`fixed_training_cell.py`**: Complete training solution
3. **`debug_data_shapes.py`**: Detailed analysis tool
4. **`memory_safe_data_loader.py`**: Memory-safe data loading
5. **`safe_data_loading_cell.py`**: Safe data loading replacement

## Usage Summary

**For immediate fix:**
```python
exec(open('quick_fix_cell.py').read())
```

**For complete solution:**
```python
exec(open('fixed_training_cell.py').read())
model, avg_train_loss, avg_valid_loss, time_per_epoch = run_fixed_training(model, config)
```

**For analysis:**
```python
exec(open('debug_data_shapes.py').read())
debug_data_shapes()
``` 