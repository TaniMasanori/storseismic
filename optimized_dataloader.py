from torch.utils.data import DataLoader

def create_optimized_dataloaders(train_data, test_data, config):
    """Create memory-optimized dataloaders"""
    
    train_dataloader = DataLoader(
        train_data, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True if config.dataloader_num_workers > 0 else False
    )
    
    test_dataloader = DataLoader(
        test_data, 
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True if config.dataloader_num_workers > 0 else False
    )
    
    return train_dataloader, test_dataloader 