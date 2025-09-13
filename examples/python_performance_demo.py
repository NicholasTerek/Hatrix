#!/usr/bin/env python3
"""
High-performance demonstration of Hatrix optimized implementations.
Shows SIMD, multi-threading, and cache optimization benefits.
"""

import numpy as np
import hatrix as hx
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

def get_performance_info():
    """Display system performance capabilities."""
    print("Hatrix Performance Demo")
    print("=" * 50)
    
    info = hx.get_performance_info()
    print("System Capabilities:")
    print(f"  AVX2: {'✓' if info['avx2_available'] else '✗'}")
    print(f"  AVX-512: {'✓' if info['avx512_available'] else '✗'}")
    print(f"  FMA: {'✓' if info['fma_available'] else '✗'}")
    print(f"  Max Threads: {info['max_threads']}")
    print(f"  Hardware Concurrency: {info['hardware_concurrency']}")
    print()

def benchmark_matrix_generation():
    """Benchmark different matrix generation methods."""
    print("Matrix Generation Performance")
    print("-" * 40)
    
    sizes = [64, 128, 256, 512, 1024]
    methods = {
        'recursive': hx.generate_recursive,
        'iterative': hx.generate_iterative,
        'optimized': lambda n: hx.create_hadamard(n, 'optimized'),
        'blocked': lambda n: hx.create_hadamard_optimized(n, 'blocked'),
        'parallel': lambda n: hx.create_hadamard_optimized(n, 'parallel'),
        'simd_parallel': lambda n: hx.create_hadamard_optimized(n, 'simd_parallel')
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n{method_name.upper()} Method:")
        results[method_name] = []
        
        for size in sizes:
            try:
                # Warmup
                method_func(min(size, 128))
                
                # Benchmark
                start_time = time.perf_counter()
                matrix = method_func(size)
                end_time = time.perf_counter()
                
                duration = (end_time - start_time) * 1000  # Convert to ms
                results[method_name].append(duration)
                
                # Verify correctness
                is_valid = hx.is_hadamard(matrix)
                print(f"  {size}x{size}: {duration:.2f} ms {'✓' if is_valid else '✗'}")
                
            except Exception as e:
                print(f"  {size}x{size}: Failed ({str(e)[:50]}...)")
                results[method_name].append(float('inf'))
    
    return results, sizes

def benchmark_fwht_transforms():
    """Benchmark different FWHT implementations."""
    print("\nFWHT Transform Performance")
    print("-" * 40)
    
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    methods = {
        'original': hx.fwht,
        'optimized': hx.fwht_optimized
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n{method_name.upper()} FWHT:")
        results[method_name] = []
        
        for size in sizes:
            try:
                # Generate test signal
                signal = np.random.randn(size).astype(np.float64)
                
                # Warmup
                if size <= 1024:
                    method_func(signal[:min(size, 512)])
                
                # Benchmark
                start_time = time.perf_counter()
                transformed = method_func(signal)
                end_time = time.perf_counter()
                
                duration = (end_time - start_time) * 1000  # Convert to ms
                results[method_name].append(duration)
                
                # Verify round-trip accuracy
                if method_name == 'optimized':
                    reconstructed = hx.ifwht_optimized(transformed)
                else:
                    reconstructed = hx.ifwht(transformed)
                
                is_exact = np.allclose(signal, reconstructed, atol=1e-10)
                print(f"  Size {size}: {duration:.2f} ms {'✓' if is_exact else '✗'}")
                
            except Exception as e:
                print(f"  Size {size}: Failed ({str(e)[:50]}...)")
                results[method_name].append(float('inf'))
    
    return results, sizes

def benchmark_matrix_operations():
    """Benchmark matrix operations."""
    print("\nMatrix Operations Performance")
    print("-" * 40)
    
    sizes = [64, 128, 256, 512]
    operations = {
        'transpose': {
            'naive': hx.transpose,
            'optimized': hx.transpose_optimized
        },
        'matrix_vector_multiply': {
            'naive': hx.multiply,
            'optimized': hx.multiply_optimized
        }
    }
    
    results = {}
    
    for op_name, methods in operations.items():
        print(f"\n{op_name.upper()}:")
        results[op_name] = {}
        
        for size in sizes:
            print(f"  Size {size}x{size}:")
            
            # Generate test data
            matrix = hx.create_hadamard(size, 'recursive')
            
            for method_name, method_func in methods.items():
                try:
                    # Prepare input
                    if 'vector' in op_name:
                        vector = np.arange(1, size + 1, dtype=np.int32)
                        # Warmup
                        if size <= 256:
                            method_func(matrix[:min(size, 128), :min(size, 128)], vector[:min(size, 128)])
                        
                        # Benchmark
                        start_time = time.perf_counter()
                        result = method_func(matrix, vector)
                        end_time = time.perf_counter()
                    else:
                        # Warmup
                        if size <= 256:
                            method_func(matrix[:min(size, 128), :min(size, 128)])
                        
                        # Benchmark
                        start_time = time.perf_counter()
                        result = method_func(matrix)
                        end_time = time.perf_counter()
                    
                    duration = (end_time - start_time) * 1000
                    
                    if op_name not in results:
                        results[op_name] = {}
                    if method_name not in results[op_name]:
                        results[op_name][method_name] = []
                    
                    results[op_name][method_name].append(duration)
                    print(f"    {method_name}: {duration:.2f} ms")
                    
                except Exception as e:
                    print(f"    {method_name}: Failed ({str(e)[:50]}...)")
                    if op_name not in results:
                        results[op_name] = {}
                    if method_name not in results[op_name]:
                        results[op_name][method_name] = []
                    results[op_name][method_name].append(float('inf'))
    
    return results, sizes

def benchmark_batch_processing():
    """Benchmark batch processing capabilities."""
    print("\nBatch Processing Performance")
    print("-" * 40)
    
    batch_configs = [
        (10, 64), (10, 256), (10, 1024),
        (100, 64), (100, 256), (100, 1024),
        (1000, 64), (1000, 256)
    ]
    
    results = {}
    
    for batch_size, signal_size in batch_configs:
        print(f"\nBatch {batch_size} x Signal {signal_size}:")
        
        # Generate test signals
        signals = np.random.randn(batch_size, signal_size).astype(np.float64)
        
        # Sequential processing
        try:
            start_time = time.perf_counter()
            sequential_results = np.zeros_like(signals)
            for i in range(batch_size):
                sequential_results[i] = hx.fwht(signals[i])
            sequential_time = (time.perf_counter() - start_time) * 1000
            
            print(f"  Sequential: {sequential_time:.2f} ms")
        except Exception as e:
            print(f"  Sequential: Failed ({str(e)[:50]}...)")
            sequential_time = float('inf')
        
        # Batch processing
        try:
            start_time = time.perf_counter()
            batch_results = hx.batch_fwht(signals)
            batch_time = (time.perf_counter() - start_time) * 1000
            
            # Verify results match
            is_correct = np.allclose(sequential_results, batch_results, atol=1e-10)
            speedup = sequential_time / batch_time if batch_time > 0 else 0
            
            print(f"  Batch: {batch_time:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x {'✓' if is_correct else '✗'}")
            
            if 'batch_speedup' not in results:
                results['batch_speedup'] = []
            results['batch_speedup'].append(speedup)
            
        except Exception as e:
            print(f"  Batch: Failed ({str(e)[:50]}...)")
    
    return results

def benchmark_scalability():
    """Benchmark scalability across different sizes."""
    print("\nScalability Analysis")
    print("-" * 30)
    
    # Matrix generation scalability
    sizes = [64, 128, 256, 512, 1024, 2048]
    methods = ['recursive', 'optimized']
    
    print("Matrix Generation Scalability:")
    for method in methods:
        print(f"\n{method.upper()}:")
        for size in sizes:
            try:
                start_time = time.perf_counter()
                if method == 'recursive':
                    matrix = hx.generate_recursive(size)
                else:
                    matrix = hx.create_hadamard(size, 'optimized')
                end_time = time.perf_counter()
                
                duration = (end_time - start_time) * 1000
                memory_mb = (size * size * 4) / (1024 * 1024)  # 4 bytes per int
                throughput = memory_mb / (duration / 1000)  # MB/s
                
                print(f"  {size}x{size}: {duration:.2f} ms, {memory_mb:.1f} MB, {throughput:.1f} MB/s")
                
            except Exception as e:
                print(f"  {size}x{size}: Failed ({str(e)[:50]}...)")
    
    # FWHT scalability
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    methods = ['original', 'optimized']
    
    print("\nFWHT Scalability:")
    for method in methods:
        print(f"\n{method.upper()}:")
        for size in sizes:
            try:
                signal = np.random.randn(size).astype(np.float64)
                
                start_time = time.perf_counter()
                if method == 'original':
                    transformed = hx.fwht(signal)
                else:
                    transformed = hx.fwht_optimized(signal)
                end_time = time.perf_counter()
                
                duration = (end_time - start_time) * 1000
                memory_mb = (size * 8 * 2) / (1024 * 1024)  # 8 bytes per double, input + output
                throughput = memory_mb / (duration / 1000)  # MB/s
                
                print(f"  Size {size}: {duration:.2f} ms, {memory_mb:.1f} MB, {throughput:.1f} MB/s")
                
            except Exception as e:
                print(f"  Size {size}: Failed ({str(e)[:50]}...)")

def create_performance_visualization():
    """Create performance visualization charts."""
    try:
        print("\nGenerating Performance Visualizations...")
        
        # Matrix generation comparison
        sizes = [64, 128, 256, 512, 1024]
        recursive_times = []
        optimized_times = []
        
        for size in sizes:
            # Recursive
            start = time.perf_counter()
            hx.generate_recursive(size)
            recursive_times.append((time.perf_counter() - start) * 1000)
            
            # Optimized
            start = time.perf_counter()
            hx.create_hadamard(size, 'optimized')
            optimized_times.append((time.perf_counter() - start) * 1000)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Matrix generation comparison
        ax1.plot(sizes, recursive_times, 'b-o', label='Recursive', linewidth=2)
        ax1.plot(sizes, optimized_times, 'r-s', label='Optimized', linewidth=2)
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Matrix Generation Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # FWHT comparison
        fwht_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        original_times = []
        optimized_times = []
        
        for size in fwht_sizes:
            signal = np.random.randn(size).astype(np.float64)
            
            # Original
            start = time.perf_counter()
            hx.fwht(signal)
            original_times.append((time.perf_counter() - start) * 1000)
            
            # Optimized
            start = time.perf_counter()
            hx.fwht_optimized(signal)
            optimized_times.append((time.perf_counter() - start) * 1000)
        
        ax2.plot(fwht_sizes, original_times, 'b-o', label='Original', linewidth=2)
        ax2.plot(fwht_sizes, optimized_times, 'r-s', label='Optimized', linewidth=2)
        ax2.set_xlabel('Signal Size')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('FWHT Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('hatrix_performance.png', dpi=150, bbox_inches='tight')
        print("Performance visualization saved to 'hatrix_performance.png'")
        
        return fig
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return None

def demonstrate_ml_features():
    """Demonstrate ML-relevant features."""
    print("\nMachine Learning Features Demo")
    print("-" * 40)
    
    # Large-scale matrix generation for ML experiments
    print("Large-scale matrix generation:")
    large_sizes = [1024, 2048]
    
    for size in large_sizes:
        try:
            print(f"\nGenerating {size}x{size} matrix...")
            start_time = time.perf_counter()
            matrix = hx.create_hadamard(size, 'optimized')
            generation_time = (time.perf_counter() - start_time) * 1000
            
            memory_mb = (size * size * 4) / (1024 * 1024)
            print(f"  Generation time: {generation_time:.2f} ms")
            print(f"  Memory usage: {memory_mb:.1f} MB")
            print(f"  Is valid Hadamard: {hx.is_hadamard(matrix)}")
            
        except Exception as e:
            print(f"  Failed: {str(e)[:50]}...")
    
    # Batch signal processing
    print("\nBatch signal processing:")
    batch_size = 1000
    signal_size = 1024
    
    try:
        # Generate random signals
        signals = np.random.randn(batch_size, signal_size).astype(np.float64)
        
        print(f"Processing {batch_size} signals of size {signal_size}...")
        start_time = time.perf_counter()
        batch_results = hx.batch_fwht(signals)
        batch_time = (time.perf_counter() - start_time) * 1000
        
        print(f"  Batch processing time: {batch_time:.2f} ms")
        print(f"  Average time per signal: {batch_time/batch_size:.3f} ms")
        print(f"  Total throughput: {batch_size/(batch_time/1000):.1f} signals/sec")
        
        # Verify correctness on a sample
        sample_idx = 0
        original_signal = signals[sample_idx]
        transformed = batch_results[sample_idx]
        reconstructed = hx.ifwht_optimized(transformed)
        
        is_exact = np.allclose(original_signal, reconstructed, atol=1e-10)
        print(f"  Round-trip accuracy: {'✓' if is_exact else '✗'}")
        
    except Exception as e:
        print(f"  Batch processing failed: {str(e)[:50]}...")

def main():
    """Main performance demonstration."""
    get_performance_info()
    
    # Run benchmarks
    matrix_results, matrix_sizes = benchmark_matrix_generation()
    fwht_results, fwht_sizes = benchmark_fwht_transforms()
    op_results, op_sizes = benchmark_matrix_operations()
    batch_results = benchmark_batch_processing()
    
    # Scalability analysis
    benchmark_scalability()
    
    # ML features demo
    demonstrate_ml_features()
    
    # Create visualizations
    fig = create_performance_visualization()
    
    print("\n" + "=" * 50)
    print("Performance Demo Completed!")
    print("=" * 50)
    
    # Summary
    print("\nKey Performance Benefits:")
    print("• SIMD vectorization for 2-8x speedup in transforms")
    print("• Multi-threading for parallel matrix operations")
    print("• Cache-optimized memory layouts for better bandwidth")
    print("• Batch processing for ML-scale experiments")
    print("• Automatic CPU feature detection and optimization")

if __name__ == "__main__":
    main()
