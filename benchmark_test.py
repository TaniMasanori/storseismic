import torch
import time
import numpy as np
from optimized_data_multiplication import multiply_data_optimized, benchmark_methods
from memory_efficient_multiplication import multiply_data_inplace, multiply_data_optimized_v2, process_datasets_parallel

def create_sample_data(batch_size=600, seq_len=20, feature_dim=271):
    """
    Create sample data that mimics the SNIST dataset structure.
    
    Args:
        batch_size: Number of samples (default: 600 for training)
        seq_len: Sequence length (default: 20)
        feature_dim: Feature dimension (default: 271)
    
    Returns:
        Dictionary with sample tensors
    """
    data_dict = {
        'inputs_embeds': torch.randn(batch_size, seq_len, feature_dim),
        'labels': torch.randn(batch_size, seq_len, feature_dim),
        'mask_label': torch.zeros(batch_size, seq_len, feature_dim),
        'index': torch.arange(batch_size)
    }
    return data_dict

def original_method(data_dict, mult_factor=10):
    """Original method from the user's code."""
    result = {key: tensor.clone() for key, tensor in data_dict.items()}
    
    for key in result.keys():
        if key != 'index':
            result[key] = result[key].repeat(mult_factor, 1, 1)
    result['index'] = torch.arange(result['inputs_embeds'].shape[0])
    
    return result

def benchmark_all_methods():
    """
    Comprehensive benchmark of all optimization methods.
    """
    print("=== データ乗算の最適化ベンチマーク ===")
    print("=== Data Multiplication Optimization Benchmark ===\n")
    
    # Create sample data
    print("Creating sample data...")
    train_data = create_sample_data(batch_size=600)  # Training data size
    test_data = create_sample_data(batch_size=150)   # Test data size
    mult_factor = 10
    num_runs = 3
    
    print(f"Training data shape: {train_data['inputs_embeds'].shape}")
    print(f"Test data shape: {test_data['inputs_embeds'].shape}")
    print(f"Multiplication factor: {mult_factor}")
    print(f"Number of benchmark runs: {num_runs}\n")
    
    methods = {
        'Original (slow)': lambda data: original_method(data, mult_factor),
        'Optimized v1': lambda data: multiply_data_optimized(data, mult_factor),
        'Optimized v2': lambda data: multiply_data_optimized_v2(data, mult_factor),
        'In-place': lambda data: multiply_data_inplace(data.copy(), mult_factor) or data  # Return data after in-place
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"Benchmarking {method_name}...")
        times = []
        
        for run in range(num_runs):
            # Test on training data
            test_data_copy = {k: v.clone() for k, v in train_data.items()}
            
            start_time = time.time()
            
            if method_name == 'In-place':
                multiply_data_inplace(test_data_copy, mult_factor)
                result = test_data_copy
            else:
                result = method_func(test_data_copy)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Verify result correctness
            expected_shape = (train_data['inputs_embeds'].shape[0] * mult_factor, 
                            train_data['inputs_embeds'].shape[1], 
                            train_data['inputs_embeds'].shape[2])
            
            if result['inputs_embeds'].shape != expected_shape:
                print(f"ERROR: {method_name} produced incorrect shape!")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results[method_name] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'times': times
        }
        
        print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds\n")
    
    # Print comparison results
    print("=== パフォーマンス比較 ===")
    print("=== Performance Comparison ===")
    
    baseline_time = results['Original (slow)']['avg_time']
    
    print(f"{'Method':<20} {'Time (s)':<12} {'Speedup':<10} {'Memory Usage'}")
    print("-" * 60)
    
    for method_name, result in results.items():
        speedup = baseline_time / result['avg_time']
        memory_usage = "Low" if 'In-place' in method_name else "Medium"
        
        print(f"{method_name:<20} {result['avg_time']:.4f}     {speedup:.2f}x      {memory_usage}")
    
    print("\n=== 推奨される使用方法 ===")
    print("=== Recommended Usage ===")
    
    best_method = min(results.items(), key=lambda x: x[1]['avg_time'])
    print(f"Fastest method: {best_method[0]} ({best_method[1]['avg_time']:.4f}s)")
    
    print("""
最適化されたコードの使用例 / Optimized code usage examples:

# Option 1: インプレース処理（最も高速、元のデータを変更）
# In-place processing (fastest, modifies original data)
multiply_data_inplace(snist_train_mlm, mult_factor=10)
multiply_data_inplace(snist_test_mlm, mult_factor=10)

# Option 2: 新しい辞書を作成（元のデータを保持）
# Create new dictionaries (preserves original data)
snist_train_mlm = multiply_data_optimized_v2(snist_train_mlm, mult_factor=10)
snist_test_mlm = multiply_data_optimized_v2(snist_test_mlm, mult_factor=10)

# Option 3: 両方を同時に処理
# Process both datasets together
snist_train_mlm, snist_test_mlm = process_datasets_parallel(
    snist_train_mlm, snist_test_mlm, mult_factor=10
)
""")

def memory_usage_test():
    """
    Test memory usage of different methods.
    """
    print("\n=== メモリ使用量テスト ===")
    print("=== Memory Usage Test ===")
    
    # Create larger sample data
    large_data = create_sample_data(batch_size=1000, seq_len=20, feature_dim=271)
    mult_factor = 10
    
    import psutil
    import os
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    print("Testing memory usage with larger dataset...")
    print(f"Data shape: {large_data['inputs_embeds'].shape}")
    
    # Test in-place method
    initial_memory = get_memory_usage()
    test_data = {k: v.clone() for k, v in large_data.items()}
    
    multiply_data_inplace(test_data, mult_factor)
    final_memory = get_memory_usage()
    
    print(f"In-place method memory increase: {final_memory - initial_memory:.2f} MB")
    
    # Test copy method
    initial_memory = get_memory_usage()
    test_data2 = {k: v.clone() for k, v in large_data.items()}
    
    result = multiply_data_optimized_v2(test_data2, mult_factor)
    final_memory = get_memory_usage()
    
    print(f"Copy method memory increase: {final_memory - initial_memory:.2f} MB")

if __name__ == "__main__":
    # Run benchmarks
    benchmark_all_methods()
    
    # Run memory tests if psutil is available
    try:
        memory_usage_test()
    except ImportError:
        print("\npsutil not available - skipping memory tests")
        print("Install with: pip install psutil") 