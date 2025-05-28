# MANUAL STEP-BY-STEP APPROACH
# Run each section in a SEPARATE CELL to isolate problems

# ============================================================================
# CELL 1: Initial setup and diagnostics
# ============================================================================
import torch
import gc
import sys

print("=== DIAGNOSTIC INFORMATION ===")
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\nCurrent data shapes:")
print(f"Training: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Test: {snist_test_mlm['inputs_embeds'].shape}")

# Check data types and devices
print(f"\nData info:")
for key in snist_train_mlm.keys():
    tensor = snist_train_mlm[key]
    print(f"  {key}: {tensor.dtype}, {tensor.device}, {tensor.shape}")

# ============================================================================
# CELL 2: Test with minimal multiplication (mult_factor = 2)
# ============================================================================
print("Testing minimal multiplication...")
mult_factor = 2

# Test with just one key first
test_tensor = snist_train_mlm['inputs_embeds']
print(f"Original tensor shape: {test_tensor.shape}")

try:
    test_result = test_tensor.repeat(mult_factor, 1, 1)
    print(f"✓ Test successful. New shape: {test_result.shape}")
    del test_result
    gc.collect()
except Exception as e:
    print(f"✗ Test failed: {e}")
    print("There may be a fundamental issue with your PyTorch installation or system")

# ============================================================================
# CELL 3: Process training inputs_embeds only
# ============================================================================
print("Processing training inputs_embeds...")
mult_factor = 10  # You can reduce this if needed

try:
    train_inputs_new = snist_train_mlm['inputs_embeds'].repeat(mult_factor, 1, 1)
    print(f"✓ Training inputs processed: {train_inputs_new.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    # Fallback to smaller factor
    mult_factor = 2
    train_inputs_new = snist_train_mlm['inputs_embeds'].repeat(mult_factor, 1, 1)
    print(f"✓ Fallback successful with mult_factor={mult_factor}: {train_inputs_new.shape}")

gc.collect()

# ============================================================================
# CELL 4: Process training labels
# ============================================================================
print("Processing training labels...")

try:
    train_labels_new = snist_train_mlm['labels'].repeat(mult_factor, 1, 1)
    print(f"✓ Training labels processed: {train_labels_new.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    train_labels_new = snist_train_mlm['labels']  # Keep original
    print("Keeping original labels")

gc.collect()

# ============================================================================
# CELL 5: Process training mask_label
# ============================================================================
print("Processing training mask_label...")

try:
    train_mask_new = snist_train_mlm['mask_label'].repeat(mult_factor, 1, 1)
    print(f"✓ Training mask processed: {train_mask_new.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    train_mask_new = snist_train_mlm['mask_label']  # Keep original
    print("Keeping original mask")

gc.collect()

# ============================================================================
# CELL 6: Reconstruct training dictionary
# ============================================================================
print("Reconstructing training dictionary...")

try:
    snist_train_mlm_new = {
        'inputs_embeds': train_inputs_new,
        'labels': train_labels_new,
        'mask_label': train_mask_new,
        'index': torch.arange(train_inputs_new.shape[0])
    }
    
    # Replace original
    snist_train_mlm = snist_train_mlm_new
    
    print(f"✓ Training data reconstructed: {snist_train_mlm['inputs_embeds'].shape}")
    
    # Clean up temporary variables
    del train_inputs_new, train_labels_new, train_mask_new, snist_train_mlm_new
    gc.collect()
    
except Exception as e:
    print(f"✗ Failed: {e}")

# ============================================================================
# CELL 7: Process test inputs_embeds
# ============================================================================
print("Processing test inputs_embeds...")

try:
    test_inputs_new = snist_test_mlm['inputs_embeds'].repeat(mult_factor, 1, 1)
    print(f"✓ Test inputs processed: {test_inputs_new.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    test_inputs_new = snist_test_mlm['inputs_embeds']  # Keep original
    print("Keeping original test inputs")

gc.collect()

# ============================================================================
# CELL 8: Process test labels
# ============================================================================
print("Processing test labels...")

try:
    test_labels_new = snist_test_mlm['labels'].repeat(mult_factor, 1, 1)
    print(f"✓ Test labels processed: {test_labels_new.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    test_labels_new = snist_test_mlm['labels']  # Keep original
    print("Keeping original test labels")

gc.collect()

# ============================================================================
# CELL 9: Process test mask_label
# ============================================================================
print("Processing test mask_label...")

try:
    test_mask_new = snist_test_mlm['mask_label'].repeat(mult_factor, 1, 1)
    print(f"✓ Test mask processed: {test_mask_new.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    test_mask_new = snist_test_mlm['mask_label']  # Keep original
    print("Keeping original test mask")

gc.collect()

# ============================================================================
# CELL 10: Reconstruct test dictionary
# ============================================================================
print("Reconstructing test dictionary...")

try:
    snist_test_mlm_new = {
        'inputs_embeds': test_inputs_new,
        'labels': test_labels_new,
        'mask_label': test_mask_new,
        'index': torch.arange(test_inputs_new.shape[0])
    }
    
    # Replace original
    snist_test_mlm = snist_test_mlm_new
    
    print(f"✓ Test data reconstructed: {snist_test_mlm['inputs_embeds'].shape}")
    
    # Clean up temporary variables
    del test_inputs_new, test_labels_new, test_mask_new, snist_test_mlm_new
    gc.collect()
    
except Exception as e:
    print(f"✗ Failed: {e}")

# ============================================================================
# CELL 11: Final verification
# ============================================================================
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Training data shape: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Test data shape: {snist_test_mlm['inputs_embeds'].shape}")
print(f"Multiplication factor used: {mult_factor}")

# Final cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Manual processing completed!")

# ============================================================================
# TROUBLESHOOTING NOTES
# ============================================================================
print("""
TROUBLESHOOTING NOTES:

If any cell crashes:
1. Note which cell crashed
2. Try reducing mult_factor to 2 or even 1
3. Check if you have enough disk space (PyTorch may use swap)
4. Try restarting the Jupyter kernel
5. Consider using a smaller dataset for testing

If all cells crash:
- There may be a PyTorch installation issue
- Try: pip install --upgrade torch
- Or reinstall PyTorch completely

If only specific tensors crash:
- Those tensors may be corrupted
- Try recreating them from the original data
""") 