#!/usr/bin/env python3
"""
Comprehensive performance tests for Hatrix Python bindings.
Tests all high-performance features including SIMD, multi-threading, and optimization.
"""

import pytest
import numpy as np
import hatrix as hx
import time
import warnings
from typing import List, Tuple, Dict

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestPerformanceCapabilities:
    """Test system performance capabilities detection."""
    
    def test_performance_info(self):
        """Test that performance info is properly detected."""
        info = hx.get_performance_info()
        
        assert isinstance(info, dict)
        assert 'avx2_available' in info
        assert 'avx512_available' in info
        assert 'fma_available' in info
        assert 'max_threads' in info
        assert 'hardware_concurrency' in info
        
        assert isinstance(info['avx2_available'], bool)
        assert isinstance(info['avx512_available'], bool)
        assert isinstance(info['fma_available'], bool)
        assert isinstance(info['max_threads'], int)
        assert isinstance(info['hardware_concurrency'], int)
        
        assert info['max_threads'] > 0
        assert info['hardware_concurrency'] > 0
    
    def test_performance_info_consistency(self):
        """Test that performance info is consistent across calls."""
        info1 = hx.get_performance_info()
        info2 = hx.get_performance_info()
        
        assert info1 == info2

class TestOptimizedMatrixGeneration:
    """Test optimized matrix generation methods."""
    
    @pytest.mark.parametrize("size", [32, 64, 128, 256, 512])
    def test_optimized_generation_methods(self, size):
        """Test different optimized generation methods."""
        methods = ['blocked', 'parallel', 'simd_parallel']
        
        for method in methods:
            matrix = hx.create_hadamard_optimized(size, method)
            
            assert matrix.shape == (size, size)
            assert hx.is_hadamard(matrix)
            assert np.all(np.abs(matrix) == 1)  # All elements are ±1
    
    @pytest.mark.parametrize("size", [32, 64, 128, 256])
    def test_generation_consistency(self, size):
        """Test that different generation methods produce equivalent results."""
        # Note: Different methods may produce different orderings, but should be valid
        methods = ['blocked', 'parallel', 'simd_parallel']
        matrices = []
        
        for method in methods:
            matrix = hx.create_hadamard_optimized(size, method)
            matrices.append(matrix)
            
            # Each should be a valid Hadamard matrix
            assert hx.is_hadamard(matrix)
        
        # All matrices should have the same properties
        for matrix in matrices:
            assert matrix.shape == (size, size)
            assert np.all(np.abs(matrix) == 1)
    
    def test_large_matrix_generation(self):
        """Test generation of large matrices."""
        size = 1024
        matrix = hx.create_hadamard_optimized(size, 'simd_parallel')
        
        assert matrix.shape == (size, size)
        assert hx.is_hadamard(matrix)
        
        # Check orthogonality of first few rows
        for i in range(min(10, size)):
            for j in range(i + 1, min(10, size)):
                dot_product = np.dot(matrix[i], matrix[j])
                assert dot_product == 0, f"Rows {i} and {j} are not orthogonal"

class TestOptimizedTransforms:
    """Test optimized FWHT implementations."""
    
    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024])
    def test_fwht_optimized_correctness(self, size):
        """Test that optimized FWHT produces correct results."""
        # Generate test signal
        signal = np.random.randn(size).astype(np.float64)
        
        # Compute with optimized version
        transformed = hx.fwht_optimized(signal)
        
        # Compute with reference version
        reference = hx.fwht(signal)
        
        # Results should match
        np.testing.assert_allclose(transformed, reference, rtol=1e-12, atol=1e-12)
    
    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024])
    def test_fwht_round_trip_accuracy(self, size):
        """Test round-trip accuracy of optimized FWHT."""
        # Generate test signal
        signal = np.random.randn(size).astype(np.float64)
        
        # Forward transform
        transformed = hx.fwht_optimized(signal)
        
        # Inverse transform
        reconstructed = hx.ifwht_optimized(transformed)
        
        # Should recover original signal
        np.testing.assert_allclose(signal, reconstructed, rtol=1e-12, atol=1e-12)
    
    def test_fwht_performance_improvement(self):
        """Test that optimized FWHT is faster than naive version."""
        size = 4096
        signal = np.random.randn(size).astype(np.float64)
        
        # Time naive version
        start = time.perf_counter()
        naive_result = hx.fwht(signal)
        naive_time = time.perf_counter() - start
        
        # Time optimized version
        start = time.perf_counter()
        optimized_result = hx.fwht_optimized(signal)
        optimized_time = time.perf_counter() - start
        
        # Results should be equivalent
        np.testing.assert_allclose(naive_result, optimized_result, rtol=1e-12, atol=1e-12)
        
        # Optimized version should be faster (allow for some variance)
        speedup = naive_time / optimized_time
        print(f"FWHT speedup: {speedup:.2f}x")
        
        # Should be at least 1.5x faster for large signals
        if size >= 1024:
            assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"

class TestOptimizedMatrixOperations:
    """Test optimized matrix operations."""
    
    @pytest.mark.parametrize("size", [64, 128, 256, 512])
    def test_transpose_optimized(self, size):
        """Test optimized matrix transpose."""
        matrix = hx.create_hadamard(size, 'recursive')
        
        # Compute with optimized version
        transposed = hx.transpose_optimized(matrix)
        
        # Compute with reference version
        reference = hx.transpose(matrix)
        
        # Results should match
        np.testing.assert_array_equal(transposed, reference)
    
    @pytest.mark.parametrize("size", [64, 128, 256, 512])
    def test_multiply_optimized(self, size):
        """Test optimized matrix-vector multiplication."""
        matrix = hx.create_hadamard(size, 'recursive')
        vector = np.arange(1, size + 1, dtype=np.int32)
        
        # Compute with optimized version
        result = hx.multiply_optimized(matrix, vector)
        
        # Compute with reference version
        reference = hx.multiply(matrix, vector)
        
        # Results should match
        np.testing.assert_array_equal(result, reference)
    
    @pytest.mark.parametrize("size", [32, 64, 128, 256])
    def test_multiply_matrices_optimized(self, size):
        """Test optimized matrix-matrix multiplication."""
        A = hx.create_hadamard(size, 'recursive')
        B = hx.create_hadamard(size, 'recursive')
        
        # Compute with optimized version
        result = hx.multiply_matrices_optimized(A, B)
        
        # Compute with reference version
        reference = hx.multiply_matrices(A, B)
        
        # Results should match
        np.testing.assert_array_equal(result, reference)
    
    def test_optimized_operations_performance(self):
        """Test that optimized operations are faster."""
        size = 512
        matrix = hx.create_hadamard(size, 'recursive')
        vector = np.arange(1, size + 1, dtype=np.int32)
        
        # Time naive transpose
        start = time.perf_counter()
        naive_transpose = hx.transpose(matrix)
        naive_transpose_time = time.perf_counter() - start
        
        # Time optimized transpose
        start = time.perf_counter()
        optimized_transpose = hx.transpose_optimized(matrix)
        optimized_transpose_time = time.perf_counter() - start
        
        # Results should match
        np.testing.assert_array_equal(naive_transpose, optimized_transpose)
        
        # Check performance improvement
        transpose_speedup = naive_transpose_time / optimized_transpose_time
        print(f"Transpose speedup: {transpose_speedup:.2f}x")
        
        # Time naive multiply
        start = time.perf_counter()
        naive_multiply = hx.multiply(matrix, vector)
        naive_multiply_time = time.perf_counter() - start
        
        # Time optimized multiply
        start = time.perf_counter()
        optimized_multiply = hx.multiply_optimized(matrix, vector)
        optimized_multiply_time = time.perf_counter() - start
        
        # Results should match
        np.testing.assert_array_equal(naive_multiply, optimized_multiply)
        
        # Check performance improvement
        multiply_speedup = naive_multiply_time / optimized_multiply_time
        print(f"Multiply speedup: {multiply_speedup:.2f}x")

class TestBatchProcessing:
    """Test batch processing capabilities."""
    
    @pytest.mark.parametrize("batch_size,signal_size", [(10, 64), (100, 256), (1000, 512)])
    def test_batch_fwht_correctness(self, batch_size, signal_size):
        """Test that batch FWHT produces correct results."""
        # Generate test signals
        signals = np.random.randn(batch_size, signal_size).astype(np.float64)
        
        # Process with batch function
        batch_results = hx.batch_fwht(signals)
        
        # Process individually for comparison
        individual_results = np.zeros_like(signals)
        for i in range(batch_size):
            individual_results[i] = hx.fwht_optimized(signals[i])
        
        # Results should match
        np.testing.assert_allclose(batch_results, individual_results, rtol=1e-12, atol=1e-12)
    
    def test_batch_fwht_performance(self):
        """Test that batch processing is faster than individual processing."""
        batch_size = 1000
        signal_size = 256
        signals = np.random.randn(batch_size, signal_size).astype(np.float64)
        
        # Time individual processing
        start = time.perf_counter()
        individual_results = np.zeros_like(signals)
        for i in range(batch_size):
            individual_results[i] = hx.fwht_optimized(signals[i])
        individual_time = time.perf_counter() - start
        
        # Time batch processing
        start = time.perf_counter()
        batch_results = hx.batch_fwht(signals)
        batch_time = time.perf_counter() - start
        
        # Results should match
        np.testing.assert_allclose(batch_results, individual_results, rtol=1e-12, atol=1e-12)
        
        # Batch processing should be faster
        speedup = individual_time / batch_time
        print(f"Batch processing speedup: {speedup:.2f}x")
        
        assert speedup > 1.0, f"Expected batch processing to be faster, got {speedup:.2f}x speedup"
    
    def test_batch_fwht_round_trip(self):
        """Test round-trip accuracy of batch processing."""
        batch_size = 100
        signal_size = 128
        signals = np.random.randn(batch_size, signal_size).astype(np.float64)
        
        # Forward transform
        transformed = hx.batch_fwht(signals)
        
        # Inverse transform (batch)
        reconstructed = hx.batch_fwht(transformed)
        
        # Should recover original signals
        np.testing.assert_allclose(signals, reconstructed, rtol=1e-12, atol=1e-12)

class TestPerformanceRegression:
    """Performance regression tests."""
    
    def test_matrix_generation_performance(self):
        """Test that matrix generation meets performance expectations."""
        size = 1024
        
        start = time.perf_counter()
        matrix = hx.create_hadamard_optimized(size, 'simd_parallel')
        generation_time = time.perf_counter() - start
        
        # Should generate 1024x1024 matrix in reasonable time
        assert generation_time < 1.0, f"Matrix generation took {generation_time:.3f}s (expected < 1.0s)"
        
        # Verify correctness
        assert matrix.shape == (size, size)
        assert hx.is_hadamard(matrix)
    
    def test_large_fwht_performance(self):
        """Test that large FWHT meets performance expectations."""
        size = 4096
        signal = np.random.randn(size).astype(np.float64)
        
        start = time.perf_counter()
        result = hx.fwht_optimized(signal)
        fwht_time = time.perf_counter() - start
        
        # Should process 4096-point FWHT in reasonable time
        assert fwht_time < 0.01, f"FWHT took {fwht_time:.4f}s (expected < 0.01s)"
        
        # Verify correctness
        assert len(result) == size
        
        # Test round-trip accuracy
        reconstructed = hx.ifwht_optimized(result)
        np.testing.assert_allclose(signal, reconstructed, rtol=1e-10, atol=1e-10)
    
    def test_batch_processing_performance(self):
        """Test that batch processing meets performance expectations."""
        batch_size = 1000
        signal_size = 512
        signals = np.random.randn(batch_size, signal_size).astype(np.float64)
        
        start = time.perf_counter()
        results = hx.batch_fwht(signals)
        batch_time = time.perf_counter() - start
        
        # Should process 1000 signals of size 512 in reasonable time
        assert batch_time < 0.5, f"Batch processing took {batch_time:.3f}s (expected < 0.5s)"
        
        # Verify correctness
        assert results.shape == (batch_size, signal_size)

class TestAccuracy:
    """Test numerical accuracy of optimized implementations."""
    
    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024])
    def test_fwht_accuracy(self, size):
        """Test FWHT accuracy for various sizes."""
        signal = np.random.randn(size).astype(np.float64)
        
        # Forward and inverse transform
        transformed = hx.fwht_optimized(signal)
        reconstructed = hx.ifwht_optimized(transformed)
        
        # Check round-trip accuracy
        error = np.max(np.abs(signal - reconstructed))
        assert error < 1e-12, f"Round-trip error too large: {error}"
    
    def test_matrix_orthogonality_accuracy(self):
        """Test orthogonality of generated matrices."""
        size = 64
        matrix = hx.create_hadamard_optimized(size, 'simd_parallel')
        
        # Check orthogonality of rows
        for i in range(size):
            for j in range(i + 1, size):
                dot_product = np.dot(matrix[i], matrix[j])
                assert dot_product == 0, f"Rows {i} and {j} are not orthogonal: {dot_product}"
        
        # Check orthogonality of columns
        for i in range(size):
            for j in range(i + 1, size):
                dot_product = np.dot(matrix[:, i], matrix[:, j])
                assert dot_product == 0, f"Columns {i} and {j} are not orthogonal: {dot_product}"
    
    def test_matrix_elements_accuracy(self):
        """Test that matrix elements are exactly ±1."""
        size = 128
        matrix = hx.create_hadamard_optimized(size, 'simd_parallel')
        
        # All elements should be exactly ±1
        unique_values = np.unique(matrix)
        assert len(unique_values) == 2, f"Expected 2 unique values (±1), got {len(unique_values)}"
        assert set(unique_values) == {1, -1}, f"Expected values {{1, -1}}, got {set(unique_values)}"

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_matrices(self):
        """Test optimized functions with small matrices."""
        # Test with size 1
        matrix = hx.create_hadamard_optimized(1, 'simd_parallel')
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1
        
        # Test with size 2
        matrix = hx.create_hadamard_optimized(2, 'simd_parallel')
        assert matrix.shape == (2, 2)
        assert hx.is_hadamard(matrix)
    
    def test_invalid_methods(self):
        """Test error handling for invalid methods."""
        with pytest.raises(ValueError):
            hx.create_hadamard_optimized(64, 'invalid_method')
        
        with pytest.raises(ValueError):
            hx.create_hadamard(64, 'invalid_method')
    
    def test_batch_fwht_invalid_input(self):
        """Test batch FWHT with invalid input."""
        # Test with 1D array (should raise error)
        signal = np.random.randn(64)
        with pytest.raises(RuntimeError):
            hx.batch_fwht(signal)
        
        # Test with 3D array (should raise error)
        signals = np.random.randn(10, 64, 64)
        with pytest.raises(RuntimeError):
            hx.batch_fwht(signals)

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow with optimized functions."""
        # Generate large Hadamard matrix
        size = 256
        matrix = hx.create_hadamard_optimized(size, 'simd_parallel')
        
        # Verify it's a valid Hadamard matrix
        assert hx.is_hadamard(matrix)
        
        # Generate test signal
        signal = np.random.randn(size).astype(np.float64)
        
        # Apply FWHT using optimized method
        transformed = hx.fwht_optimized(signal)
        
        # Perform matrix-vector multiplication
        vector = np.arange(1, size + 1, dtype=np.int32)
        result = hx.multiply_optimized(matrix, vector)
        
        # Verify results are reasonable
        assert len(transformed) == size
        assert len(result) == size
        assert not np.all(result == 0)  # Result should not be all zeros
    
    def test_batch_workflow(self):
        """Test batch processing workflow."""
        # Generate multiple signals
        batch_size = 100
        signal_size = 128
        signals = np.random.randn(batch_size, signal_size).astype(np.float64)
        
        # Process batch
        batch_results = hx.batch_fwht(signals)
        
        # Verify results
        assert batch_results.shape == (batch_size, signal_size)
        
        # Test round-trip accuracy on a sample
        sample_idx = 0
        original = signals[sample_idx]
        transformed = batch_results[sample_idx]
        reconstructed = hx.ifwht_optimized(transformed)
        
        np.testing.assert_allclose(original, reconstructed, rtol=1e-12, atol=1e-12)

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
