#!/usr/bin/env python3
"""
Basic usage examples for the Hatrix Python bindings.
Demonstrates core functionality with simple examples.
"""

import numpy as np
import hatrix as hx
import matplotlib.pyplot as plt

def main():
    print("Hatrix Python Library - Basic Usage Examples")
    print("=" * 50)
    
    #--------------------------------------------------------------------------
    # Example 1: Generate and Display Hadamard Matrices
    #--------------------------------------------------------------------------
    print("\n1. Matrix Generation")
    print("-" * 20)
    
    # Generate different sized matrices
    H2 = hx.generate_recursive(2)
    H4 = hx.generate_recursive(4)
    H8 = hx.generate_iterative(8)
    
    print("H(2):")
    print(H2)
    print("\nH(4):")
    print(H4)
    print("\nH(8) shape:", H8.shape)
    
    #--------------------------------------------------------------------------
    # Example 2: Walsh Orderings
    #--------------------------------------------------------------------------
    print("\n2. Walsh Orderings")
    print("-" * 20)
    
    W_natural = hx.generate_walsh(4, hx.Ordering.NATURAL)
    W_sequency = hx.generate_walsh(4, hx.Ordering.SEQUENCY)
    W_dyadic = hx.generate_walsh(4, hx.Ordering.DYADIC)
    
    print("Natural ordering:")
    print(W_natural)
    print("\nSequency ordering:")
    print(W_sequency)
    print("\nDyadic ordering:")
    print(W_dyadic)
    
    #--------------------------------------------------------------------------
    # Example 3: Matrix Validation
    #--------------------------------------------------------------------------
    print("\n3. Matrix Validation")
    print("-" * 20)
    
    # Validate matrices
    print("H(4) is Hadamard:", hx.is_hadamard(H4))
    print("H(4) is orthogonal:", hx.is_orthogonal(H4))
    
    # Check for issues
    issues = hx.validate_matrix(H4)
    if not issues:
        print("No validation issues found")
    else:
        print("Validation issues:", issues)
    
    # Test invalid matrix
    invalid_matrix = np.array([[1, 1], [1, 2]])  # Contains 2, not ±1
    print("Invalid matrix is Hadamard:", hx.is_hadamard(invalid_matrix))
    
    #--------------------------------------------------------------------------
    # Example 4: Matrix Operations
    #--------------------------------------------------------------------------
    print("\n4. Matrix Operations")
    print("-" * 20)
    
    # Transpose
    H4_T = hx.transpose(H4)
    print("H(4) transpose:")
    print(H4_T)
    
    # Matrix-vector multiplication
    v = np.array([1, 2, 3, 4], dtype=np.int32)
    result = hx.multiply(H4, v)
    print("\nH(4) * [1,2,3,4] =", result)
    
    # Matrix-matrix multiplication (should give n*I)
    H4H4T = hx.multiply_matrices(H4, H4_T)
    print("\nH(4) * H(4)^T (should be 4*I):")
    print(H4H4T)
    
    #--------------------------------------------------------------------------
    # Example 5: Fast Walsh-Hadamard Transform
    #--------------------------------------------------------------------------
    print("\n5. Fast Walsh-Hadamard Transform")
    print("-" * 30)
    
    # Create test signal
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    print("Original signal:", signal)
    
    # Forward transform
    transformed = hx.fwht(signal)
    print("Transformed:", transformed)
    
    # Inverse transform
    reconstructed = hx.ifwht(transformed)
    print("Reconstructed:", reconstructed)
    
    # Check round-trip accuracy
    is_exact = np.allclose(signal, reconstructed)
    print("Round-trip exact:", is_exact)
    
    #--------------------------------------------------------------------------
    # Example 6: Signal Processing
    #--------------------------------------------------------------------------
    print("\n6. Signal Processing Example")
    print("-" * 30)
    
    # Create signal with noise
    np.random.seed(42)
    clean_signal = np.sin(2 * np.pi * np.arange(32) / 32) + 0.5 * np.sin(4 * np.pi * np.arange(32) / 32)
    noise = np.random.normal(0, 0.1, 32)
    noisy_signal = clean_signal + noise
    
    # Apply FWHT
    clean_fwht = hx.fwht(clean_signal)
    noisy_fwht = hx.fwht(noisy_signal)
    
    # Find significant coefficients
    threshold = 0.1
    significant_mask = np.abs(clean_fwht) > threshold
    significant_count = np.sum(significant_mask)
    
    print(f"Signal size: {len(clean_signal)}")
    print(f"Significant coefficients (|coeff| > {threshold}): {significant_count}")
    
    # Noise filtering: zero out small coefficients
    filtered_fwht = noisy_fwht.copy()
    filtered_fwht[np.abs(filtered_fwht) < threshold] = 0.0
    filtered_signal = hx.ifwht(filtered_fwht)
    
    # Calculate SNR improvement
    original_noise_power = np.mean((noisy_signal - clean_signal) ** 2)
    filtered_noise_power = np.mean((filtered_signal - clean_signal) ** 2)
    snr_improvement = 10 * np.log10(original_noise_power / filtered_noise_power)
    
    print(f"SNR improvement: {snr_improvement:.2f} dB")
    
    #--------------------------------------------------------------------------
    # Example 7: Matrix Properties Analysis
    #--------------------------------------------------------------------------
    print("\n7. Matrix Properties Analysis")
    print("-" * 30)
    
    props = hx.analyze_properties(H4)
    print("Matrix properties:")
    for key, value in props.items():
        print(f"  {key}: {value:.3f}")
    
    #--------------------------------------------------------------------------
    # Example 8: Serialization
    #--------------------------------------------------------------------------
    print("\n8. Serialization")
    print("-" * 15)
    
    # Serialize to different formats
    compact_data = hx.serialize(H4, hx.Format.COMPACT)
    csv_data = hx.serialize(H4, hx.Format.CSV)
    
    print("Compact format:")
    print(compact_data)
    print("\nCSV format:")
    print(csv_data)
    
    # Deserialize and verify
    deserialized = hx.deserialize(compact_data, hx.Format.COMPACT)
    is_same = np.array_equal(H4, deserialized)
    print(f"\nDeserialized matrix matches original: {is_same}")
    
    #--------------------------------------------------------------------------
    # Example 9: File I/O
    #--------------------------------------------------------------------------
    print("\n9. File I/O")
    print("-" * 10)
    
    try:
        # Save to file
        hx.save_to_file(H4, "hadamard_4.txt", hx.Format.COMPACT)
        print("Matrix saved to hadamard_4.txt")
        
        # Load from file
        loaded = hx.load_from_file("hadamard_4.txt", hx.Format.COMPACT)
        is_same = np.array_equal(H4, loaded)
        print(f"Matrix loaded from file matches original: {is_same}")
        
        # Clean up
        import os
        os.remove("hadamard_4.txt")
        
    except Exception as e:
        print(f"File I/O error: {e}")
    
    #--------------------------------------------------------------------------
    # Example 10: Convenience Functions
    #--------------------------------------------------------------------------
    print("\n10. Convenience Functions")
    print("-" * 25)
    
    # Using convenience functions
    H_convenient = hx.create_hadamard(4, "recursive")
    W_convenient = hx.create_walsh(4, "sequency")
    
    print("Created with convenience functions:")
    print("H(4):", H_convenient.shape)
    print("W(4) sequency:", W_convenient.shape)
    
    #--------------------------------------------------------------------------
    # Example 11: Benchmarking
    #--------------------------------------------------------------------------
    print("\n11. Benchmarking")
    print("-" * 15)
    
    # Benchmark matrix generation
    def generate_test():
        H = hx.generate_recursive(16)
        return H
    
    time_ms = hx.benchmark(generate_test, 100) / 1000.0  # Convert to ms
    print(f"Generation time (H(16), 100 iterations): {time_ms:.2f} ms")
    
    # Benchmark transform
    test_signal = np.arange(1, 17, dtype=np.float64)
    def transform_test():
        return hx.fwht(test_signal)
    
    fwht_time_ms = hx.benchmark(transform_test, 100) / 1000.0
    print(f"FWHT time (size 16, 100 iterations): {fwht_time_ms:.2f} ms")
    
    print("\nBasic usage examples completed!")

if __name__ == "__main__":
    main()
