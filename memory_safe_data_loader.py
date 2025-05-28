import torch
import os
import gc
import psutil
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any
import warnings

class MemorySafeDataLoader:
    """
    A memory-safe data loader that can handle large PyTorch data files
    without causing kernel crashes due to memory overflow.
    """
    
    def __init__(self, data_path: str, batch_size: int = 32, max_memory_gb: float = 8.0):
        """
        Initialize the memory-safe data loader.
        
        Args:
            data_path: Path to the data directory
            batch_size: Batch size for data loading
            max_memory_gb: Maximum memory usage in GB before triggering cleanup
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.train_data = None
        self.test_data = None
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    
    def check_memory_and_cleanup(self):
        """Check memory usage and cleanup if necessary."""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_gb:
            print(f"Memory usage ({current_memory:.2f}GB) exceeds limit ({self.max_memory_gb}GB). Cleaning up...")
            torch.cuda.empty_cache()
            gc.collect()
            
    def load_data_safely(self, filename: str, chunk_size: Optional[int] = None) -> Any:
        """
        Safely load PyTorch data file with memory monitoring.
        
        Args:
            filename: Name of the data file
            chunk_size: If specified, load data in chunks
            
        Returns:
            Loaded data or None if loading fails
        """
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        file_size_gb = os.path.getsize(filepath) / 1024 / 1024 / 1024
        current_memory = self.get_memory_usage()
        
        print(f"File size: {file_size_gb:.2f}GB")
        print(f"Current memory usage: {current_memory:.2f}GB")
        
        # Check if we have enough memory
        if current_memory + file_size_gb > self.max_memory_gb:
            warnings.warn(
                f"Loading {filename} ({file_size_gb:.2f}GB) may exceed memory limit. "
                f"Consider using a smaller batch size or increasing max_memory_gb."
            )
            
        try:
            # Clear cache before loading
            self.check_memory_and_cleanup()
            
            print(f"Loading {filename}...")
            data = torch.load(filepath, map_location='cpu', weights_only=False)
            
            print(f"Successfully loaded {filename}")
            print(f"Memory usage after loading: {self.get_memory_usage():.2f}GB")
            
            return data
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            self.check_memory_and_cleanup()
            return None
    
    def create_data_loaders(self, train_filename: str = 'train_data.pt', 
                           test_filename: str = 'test_data.pt') -> tuple:
        """
        Create PyTorch DataLoaders with memory safety.
        
        Args:
            train_filename: Name of training data file
            test_filename: Name of test data file
            
        Returns:
            Tuple of (train_dataloader, test_dataloader)
        """
        print("Creating memory-safe data loaders...")
        
        # Load training data
        print("\n=== Loading Training Data ===")
        self.train_data = self.load_data_safely(train_filename)
        
        if self.train_data is None:
            raise RuntimeError("Failed to load training data")
            
        # Load test data
        print("\n=== Loading Test Data ===")
        self.test_data = self.load_data_safely(test_filename)
        
        if self.test_data is None:
            raise RuntimeError("Failed to load test data")
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=False,  # Disable pin_memory to save GPU memory
            num_workers=0      # Use single process to avoid memory duplication
        )
        
        test_dataloader = DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=False,
            num_workers=0
        )
        
        print(f"\nDataLoaders created successfully!")
        print(f"Training batches: {len(train_dataloader)}")
        print(f"Test batches: {len(test_dataloader)}")
        print(f"Final memory usage: {self.get_memory_usage():.2f}GB")
        
        return train_dataloader, test_dataloader
    
    def get_sample_batch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Get a sample batch for inspection."""
        for batch in dataloader:
            return batch
        return None
    
    def cleanup(self):
        """Clean up loaded data and free memory."""
        print("Cleaning up data loaders...")
        self.train_data = None
        self.test_data = None
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Memory usage after cleanup: {self.get_memory_usage():.2f}GB")

def create_safe_data_loaders(data_path: str, batch_size: int = 256, 
                           max_memory_gb: float = 8.0) -> tuple:
    """
    Convenience function to create memory-safe data loaders.
    
    Args:
        data_path: Path to the data directory
        batch_size: Batch size for data loading
        max_memory_gb: Maximum memory usage in GB
        
    Returns:
        Tuple of (train_dataloader, test_dataloader, loader_instance)
    """
    loader = MemorySafeDataLoader(data_path, batch_size, max_memory_gb)
    train_dl, test_dl = loader.create_data_loaders()
    return train_dl, test_dl, loader

# Example usage for the notebook
if __name__ == "__main__":
    # This can be used directly in the notebook
    data_path = './data/pretrain/'
    batch_size = 256
    
    # Create memory-safe data loaders
    train_dataloader, test_dataloader, loader = create_safe_data_loaders(
        data_path, batch_size, max_memory_gb=12.0
    )
    
    # Get sample batch for visualization
    sample_batch = loader.get_sample_batch(train_dataloader)
    print(f"Sample batch keys: {sample_batch.keys() if sample_batch else 'None'}") 