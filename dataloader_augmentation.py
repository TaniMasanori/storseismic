# DATALOADER-BASED AUGMENTATION SOLUTION
# Apply data multiplication during training instead of preprocessing

import torch
import torch.utils.data as data
import numpy as np

class AugmentedSNISTDataset(data.Dataset):
    """
    Dataset that applies augmentation on-the-fly during training.
    This prevents memory issues by not storing all augmented data at once.
    """
    
    def __init__(self, data_dict, mult_factor=10, enable_shifts=True, enable_polarity=True):
        """
        Args:
            data_dict: Original SNIST data dictionary
            mult_factor: How many times to multiply the data
            enable_shifts: Whether to apply time shifts
            enable_polarity: Whether to apply polarity reversal
        """
        self.data_dict = data_dict
        self.mult_factor = mult_factor
        self.enable_shifts = enable_shifts
        self.enable_polarity = enable_polarity
        
        # Original dataset size
        self.original_size = data_dict['inputs_embeds'].shape[0]
        
        # Effective dataset size after augmentation
        polarity_mult = 2 if enable_polarity else 1
        shift_mult = 3 if enable_shifts else 1  # original + 2 shifts
        self.total_size = self.original_size * mult_factor * shift_mult * polarity_mult
        
        print(f"Original dataset size: {self.original_size}")
        print(f"Augmented dataset size: {self.total_size}")
        print(f"Augmentation factors: mult={mult_factor}, shifts={shift_mult}, polarity={polarity_mult}")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """
        Get an item with on-the-fly augmentation.
        """
        # Map the augmented index back to original index and augmentation parameters
        original_idx, aug_params = self._decode_index(idx)
        
        # Get original data
        item = {}
        for key in self.data_dict.keys():
            if key != 'index':
                item[key] = self.data_dict[key][original_idx].clone()
        
        # Apply augmentations based on parameters
        item = self._apply_augmentations(item, aug_params)
        
        # Add the augmented index
        item['index'] = torch.tensor(idx)
        
        return item
    
    def _decode_index(self, idx):
        """
        Decode the augmented index to get original index and augmentation parameters.
        """
        # Calculate which augmentation this index corresponds to
        polarity_mult = 2 if self.enable_polarity else 1
        shift_mult = 3 if self.enable_shifts else 1
        
        # Decode the index
        remaining = idx
        
        # Polarity (outermost)
        polarity_flip = False
        if self.enable_polarity:
            polarity_flip = (remaining // (self.original_size * self.mult_factor * shift_mult)) % 2 == 1
            remaining = remaining % (self.original_size * self.mult_factor * shift_mult)
        
        # Time shifts
        shift_type = 0  # 0=no shift, 1=shift1, 2=shift2
        if self.enable_shifts:
            shift_type = (remaining // (self.original_size * self.mult_factor)) % shift_mult
            remaining = remaining % (self.original_size * self.mult_factor)
        
        # Multiplication (innermost)
        mult_idx = remaining // self.original_size
        original_idx = remaining % self.original_size
        
        aug_params = {
            'polarity_flip': polarity_flip,
            'shift_type': shift_type,
            'mult_idx': mult_idx
        }
        
        return original_idx, aug_params
    
    def _apply_augmentations(self, item, aug_params):
        """
        Apply augmentations based on parameters.
        """
        # Apply time shifts
        if self.enable_shifts and aug_params['shift_type'] > 0:
            item = self._apply_time_shift(item, aug_params['shift_type'], aug_params['mult_idx'])
        
        # Apply polarity reversal
        if self.enable_polarity and aug_params['polarity_flip']:
            item = self._apply_polarity_flip(item)
        
        return item
    
    def _apply_time_shift(self, item, shift_type, seed):
        """
        Apply time shift augmentation.
        """
        # Use seed for reproducible shifts
        torch.manual_seed(seed + shift_type * 1000)
        
        # Generate shift parameters
        min_shift, max_shift = -15, 15  # Reduced range for stability
        shift_mag = 0
        while shift_mag == 0:
            shift_mag = torch.randint(min_shift, max_shift + 1, (1,)).item()
        
        # Calculate filler value
        filler = torch.mean(item['inputs_embeds']).item()
        
        # Apply shift to inputs_embeds and labels
        for key in ['inputs_embeds', 'labels']:
            if key in item:
                # Apply roll
                item[key] = torch.roll(item[key], shift_mag, dim=-1)
                
                # Fill edges
                if shift_mag > 0:
                    item[key][:, :shift_mag] = filler
                elif shift_mag < 0:
                    item[key][:, item[key].shape[-1] + shift_mag:] = filler
        
        return item
    
    def _apply_polarity_flip(self, item):
        """
        Apply polarity reversal.
        """
        for key in ['inputs_embeds', 'labels']:
            if key in item:
                item[key] = item[key] * -1
        
        return item

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def create_augmented_dataloaders(snist_train_mlm, snist_test_mlm, 
                                mult_factor=10, batch_size=32,
                                enable_shifts=True, enable_polarity=True):
    """
    Create augmented dataloaders that apply augmentation on-the-fly.
    
    Args:
        snist_train_mlm: Training data dictionary
        snist_test_mlm: Test data dictionary
        mult_factor: Data multiplication factor
        batch_size: Batch size for dataloaders
        enable_shifts: Whether to apply time shifts
        enable_polarity: Whether to apply polarity reversal
    
    Returns:
        train_dataloader, test_dataloader
    """
    print("Creating augmented datasets...")
    
    # Create augmented datasets
    train_dataset = AugmentedSNISTDataset(
        snist_train_mlm, 
        mult_factor=mult_factor,
        enable_shifts=enable_shifts,
        enable_polarity=enable_polarity
    )
    
    test_dataset = AugmentedSNISTDataset(
        snist_test_mlm,
        mult_factor=mult_factor,
        enable_shifts=enable_shifts,
        enable_polarity=enable_polarity
    )
    
    # Create dataloaders
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✓ Training dataloader: {len(train_dataloader)} batches")
    print(f"✓ Test dataloader: {len(test_dataloader)} batches")
    
    return train_dataloader, test_dataloader

# ============================================================================
# SIMPLE USAGE - REPLACE YOUR PROBLEMATIC CELLS WITH THIS
# ============================================================================

print("🚀 DATALOADER-BASED AUGMENTATION SOLUTION 🚀")
print("This applies augmentation during training, not preprocessing")

# Configuration
mult_factor = 10
batch_size = 32
enable_shifts = True      # Set to False if you want to skip time shifts
enable_polarity = True    # Set to False if you want to skip polarity reversal

# Create augmented dataloaders
train_dataloader, test_dataloader = create_augmented_dataloaders(
    snist_train_mlm, 
    snist_test_mlm,
    mult_factor=mult_factor,
    batch_size=batch_size,
    enable_shifts=enable_shifts,
    enable_polarity=enable_polarity
)

print("\n✅ SUCCESS! Augmented dataloaders created without memory issues")
print("✅ Use these dataloaders in your training loop")
print("✅ No need to modify snist_train_mlm or snist_test_mlm")

# Test the dataloader
print("\n=== TESTING DATALOADER ===")
try:
    # Get a sample batch
    sample_batch = next(iter(train_dataloader))
    
    print(f"Sample batch shapes:")
    for key, tensor in sample_batch.items():
        print(f"  {key}: {tensor.shape}")
    
    print("✓ Dataloader test successful!")
    
except Exception as e:
    print(f"✗ Dataloader test failed: {e}")

# ============================================================================
# INTEGRATION WITH EXISTING TRAINING CODE
# ============================================================================

print("""
=== INTEGRATION INSTRUCTIONS ===

Replace your existing DataLoader creation with:

# OLD CODE (remove this):
# train_data = utils.SSDataset(snist_train_mlm)
# test_data = utils.SSDataset(snist_test_mlm)
# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# NEW CODE (use this instead):
train_dataloader, test_dataloader = create_augmented_dataloaders(
    snist_train_mlm, snist_test_mlm, mult_factor=10, batch_size=32
)

# Your training loop remains the same!
# The augmentation happens automatically during training.
""")

print("\n🎯 READY FOR TRAINING! 🎯")
print("Your training loop can now use train_dataloader and test_dataloader normally.") 