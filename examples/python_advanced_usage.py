#!/usr/bin/env python3
"""
Advanced usage examples for the Hatrix Python bindings.
Demonstrates complex scenarios, signal processing, and optimization techniques.
"""

import numpy as np
import hatrix as hx
import matplotlib.pyplot as plt
import time
from typing import List, Tuple

def benchmark_generation_methods():
    """Compare recursive vs iterative generation performance."""
    print("Generation Method Comparison")
    print("=" * 35)
    
    sizes = [8, 16, 32, 64, 128, 256, 512]
    
    print("Size\tRecursive(ms)\tIterative(ms)\tRatio")
    print("-" * 50)
    
    for size in sizes:
        # Benchmark recursive
        start = time.time()
        for _ in range(10):
            H_rec = hx.generate_recursive(size)
        rec_time = (time.time() - start) * 1000 / 10
        
        # Benchmark iterative
        start = time.time()
        for _ in range(10):
            H_iter = hx.generate_iterative(size)
        iter_time = (time.time() - start) * 1000 / 10
        
        ratio = rec_time / iter_time if iter_time > 0 else 0
        
        print(f"{size}\t{rec_time:.2f}\t\t{iter_time:.2f}\t\t{ratio:.2f}")

def large_matrix_analysis():
    """Analyze properties of large Hadamard matrices."""
    print("\nLarge Matrix Analysis")
    print("=" * 25)
    
    sizes = [64, 128, 256, 512]
    
    for size in sizes:
        print(f"\nAnalyzing H({size}):")
        
        # Generate matrix
        start_time = time.time()
        H = hx.generate_iterative(size)
        gen_time = time.time() - start_time
        
        # Analyze properties
        start_time = time.time()
        props = hx.analyze_properties(H)
        analysis_time = time.time() - start_time
        
        # Memory usage
        memory_mb = (size * size * 4) / (1024 * 1024)  # 4 bytes per int
        
        print(f"  Generation time: {gen_time:.3f} s")
        print(f"  Analysis time: {analysis_time:.3f} s")
        print(f"  Memory usage: {memory_mb:.2f} MB")
        print(f"  Is valid Hadamard: {props['is_hadamard'] == 1.0}")
        print(f"  Average sequency: {props['average_sequency']:.2f}")

def signal_processing_demo():
    """Demonstrate advanced signal processing with FWHT."""
    print("\nAdvanced Signal Processing")
    print("=" * 30)
    
    # Create a complex signal with multiple frequency components
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    
    # Signal with multiple frequency components
    signal = (np.sin(2 * np.pi * 5 * t) + 
              0.5 * np.sin(2 * np.pi * 15 * t) + 
              0.3 * np.sin(2 * np.pi * 35 * t) +
              0.2 * np.sin(2 * np.pi * 80 * t))
    
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, n)
    noisy_signal = signal + noise
    
    # Apply FWHT
    signal_fwht = hx.fwht(signal)
    noisy_fwht = hx.fwht(noisy_signal)
    
    # Frequency domain analysis
    threshold = np.std(noisy_fwht) * 2  # 2-sigma threshold
    significant_coeffs = np.abs(noisy_fwht) > threshold
    
    print(f"Signal size: {n}")
    print(f"Significant coefficients: {np.sum(significant_coeffs)} / {n}")
    print(f"Compression ratio: {np.sum(significant_coeffs) / n:.2%}")
    
    # Denoising by thresholding
    denoised_fwht = noisy_fwht.copy()
    denoised_fwht[~significant_coeffs] = 0.0
    denoised_signal = hx.ifwht(denoised_fwht)
    
    # Calculate metrics
    original_snr = 10 * np.log10(np.var(signal) / np.var(noise))
    denoised_noise = denoised_signal - signal
    denoised_snr = 10 * np.log10(np.var(signal) / np.var(denoised_noise))
    snr_improvement = denoised_snr - original_snr
    
    print(f"Original SNR: {original_snr:.2f} dB")
    print(f"Denoised SNR: {denoised_snr:.2f} dB")
    print(f"SNR improvement: {snr_improvement:.2f} dB")
    
    return signal, noisy_signal, denoised_signal

def batch_processing_demo():
    """Demonstrate batch processing of multiple signals."""
    print("\nBatch Processing Demo")
    print("=" * 25)
    
    batch_size = 100
    signal_length = 128
    
    # Generate random signals
    np.random.seed(42)
    signals = np.random.randn(batch_size, signal_length)
    
    # Process batch
    start_time = time.time()
    transformed_signals = np.zeros_like(signals)
    
    for i in range(batch_size):
        transformed_signals[i] = hx.fwht(signals[i])
    
    batch_time = time.time() - start_time
    
    # Verify round-trip accuracy
    reconstructed_signals = np.zeros_like(signals)
    for i in range(batch_size):
        reconstructed_signals[i] = hx.ifwht(transformed_signals[i])
    
    roundtrip_errors = np.abs(signals - reconstructed_signals)
    max_error = np.max(roundtrip_errors)
    mean_error = np.mean(roundtrip_errors)
    
    print(f"Processed {batch_size} signals of length {signal_length}")
    print(f"Total time: {batch_time:.3f} s")
    print(f"Average time per signal: {batch_time/batch_size*1000:.2f} ms")
    print(f"Max round-trip error: {max_error:.2e}")
    print(f"Mean round-trip error: {mean_error:.2e}")
    print(f"All round-trips correct: {max_error < 1e-10}")

def matrix_decomposition_demo():
    """Demonstrate matrix decomposition and reconstruction."""
    print("\nMatrix Decomposition Demo")
    print("=" * 30)
    
    # Generate H(8)
    H8 = hx.generate_recursive(8)
    
    # Decompose into 4 submatrices
    half = 4
    H00 = H8[:half, :half]
    H01 = H8[:half, half:]
    H10 = H8[half:, :half]
    H11 = H8[half:, half:]
    
    print("H(8) decomposed into 4 submatrices of size 4x4")
    print("Top-left submatrix:")
    print(H00)
    
    # Verify Sylvester construction properties
    print(f"\nSylvester construction verification:")
    print(f"H[0:4,0:4] == H[0:4,4:8]: {np.array_equal(H00, H01)}")
    print(f"H[0:4,0:4] == H[4:8,0:4]: {np.array_equal(H00, H10)}")
    print(f"H[4:8,4:8] == -H[0:4,0:4]: {np.array_equal(H11, -H00)}")
    
    # Reconstruct from submatrices
    reconstructed = np.zeros_like(H8)
    reconstructed[:half, :half] = H00
    reconstructed[:half, half:] = H01
    reconstructed[half:, :half] = H10
    reconstructed[half:, half:] = H11
    
    print(f"Reconstruction matches original: {np.array_equal(H8, reconstructed)}")

def walsh_ordering_analysis():
    """Analyze different Walsh orderings."""
    print("\nWalsh Ordering Analysis")
    print("=" * 25)
    
    size = 8
    orderings = [
        ("Natural", hx.Ordering.NATURAL),
        ("Sequency", hx.Ordering.SEQUENCY),
        ("Dyadic", hx.Ordering.DYADIC)
    ]
    
    for name, ordering in orderings:
        W = hx.generate_walsh(size, ordering)
        
        # Calculate sequency (number of sign changes) for each row
        sequencies = []
        for i in range(size):
            changes = np.sum(W[i, 1:] != W[i, :-1])
            sequencies.append(changes)
        
        avg_sequency = np.mean(sequencies)
        std_sequency = np.std(sequencies)
        
        print(f"\n{name} ordering:")
        print(f"  Average sequency: {avg_sequency:.2f} ± {std_sequency:.2f}")
        print(f"  Sequency range: {min(sequencies)} - {max(sequencies)}")
        print(f"  Matrix:\n{W}")

def performance_scaling_analysis():
    """Analyze performance scaling with matrix size."""
    print("\nPerformance Scaling Analysis")
    print("=" * 35)
    
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    
    print("Size\tGen(ms)\tTrans(ms)\tMemory(MB)")
    print("-" * 45)
    
    for size in sizes:
        # Generation time
        start = time.time()
        H = hx.generate_iterative(size)
        gen_time = (time.time() - start) * 1000
        
        # Transform time
        test_signal = np.arange(size, dtype=np.float64)
        start = time.time()
        hx.fwht(test_signal)
        trans_time = (time.time() - start) * 1000
        
        # Memory usage
        memory_mb = (size * size * 4) / (1024 * 1024)
        
        print(f"{size}\t{gen_time:.2f}\t{trans_time:.3f}\t\t{memory_mb:.2f}")

def error_analysis():
    """Analyze numerical errors in transforms."""
    print("\nNumerical Error Analysis")
    print("=" * 30)
    
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    
    print("Size\tMax Error\tMean Error\tMax Rel Error")
    print("-" * 50)
    
    for size in sizes:
        # Test signal
        signal = np.random.randn(size).astype(np.float64)
        
        # Forward and inverse transform
        transformed = hx.fwht(signal)
        reconstructed = hx.ifwht(transformed)
        
        # Error analysis
        absolute_errors = np.abs(signal - reconstructed)
        relative_errors = absolute_errors / (np.abs(signal) + 1e-15)
        
        max_error = np.max(absolute_errors)
        mean_error = np.mean(absolute_errors)
        max_rel_error = np.max(relative_errors)
        
        print(f"{size}\t{max_error:.2e}\t{mean_error:.2e}\t{max_rel_error:.2e}")

def visualization_demo():
    """Create visualizations of Hadamard matrices and transforms."""
    print("\nVisualization Demo")
    print("=" * 20)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Generate matrices for visualization
    H8 = hx.generate_recursive(8)
    W8_nat = hx.generate_walsh(8, hx.Ordering.NATURAL)
    W8_seq = hx.generate_walsh(8, hx.Ordering.SEQUENCY)
    
    # Plot matrices
    im1 = axes[0, 0].imshow(H8, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 0].set_title('H(8) - Natural')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(W8_nat, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('W(8) - Natural Ordering')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(W8_seq, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 2].set_title('W(8) - Sequency Ordering')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot transform example
    signal = np.sin(2 * np.pi * np.arange(64) / 64) + 0.5 * np.sin(4 * np.pi * np.arange(64) / 64)
    transformed = hx.fwht(signal)
    
    axes[1, 0].plot(signal)
    axes[1, 0].set_title('Original Signal')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(np.abs(transformed))
    axes[1, 1].set_title('FWHT Magnitude')
    axes[1, 1].set_xlabel('Coefficient')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True)
    
    # Round-trip test
    reconstructed = hx.ifwht(transformed)
    axes[1, 2].plot(signal, label='Original', alpha=0.7)
    axes[1, 2].plot(reconstructed, '--', label='Reconstructed', alpha=0.7)
    axes[1, 2].set_title('Round-trip Test')
    axes[1, 2].set_xlabel('Sample')
    axes[1, 2].set_ylabel('Amplitude')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('hatrix_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'hatrix_visualization.png'")
    
    return fig

def main():
    print("Hatrix Python Library - Advanced Usage Examples")
    print("=" * 55)
    
    try:
        benchmark_generation_methods()
        large_matrix_analysis()
        signal_processing_demo()
        batch_processing_demo()
        matrix_decomposition_demo()
        walsh_ordering_analysis()
        performance_scaling_analysis()
        error_analysis()
        
        # Visualization (optional, requires matplotlib)
        try:
            fig = visualization_demo()
            plt.show()
        except ImportError:
            print("\nNote: matplotlib not available, skipping visualization")
        
        print("\nAdvanced usage examples completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
