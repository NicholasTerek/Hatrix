#!/usr/bin/env python3
"""
Python unit tests for the Hatrix library using pytest.
"""

import pytest
import numpy as np
import hatrix as hx
import tempfile
import os


class TestHadamardMatrix:
    """Test Hadamard matrix generation and validation."""
    
    def test_generate_recursive_basic(self):
        """Test basic recursive generation."""
        H1 = hx.generate_recursive(1)
        assert H1.shape == (1, 1)
        assert H1[0, 0] == 1
        
        H2 = hx.generate_recursive(2)
        assert H2.shape == (2, 2)
        assert np.array_equal(H2, np.array([[1, 1], [1, -1]]))
        
        H4 = hx.generate_recursive(4)
        assert H4.shape == (4, 4)
    
    def test_generate_iterative_matches_recursive(self):
        """Test that iterative matches recursive for various sizes."""
        sizes = [1, 2, 4, 8, 16, 32, 64]
        
        for size in sizes:
            H_rec = hx.generate_recursive(size)
            H_iter = hx.generate_iterative(size)
            assert H_rec.shape == H_iter.shape
            assert np.array_equal(H_rec, H_iter)
    
    def test_generate_walsh_orderings(self):
        """Test different Walsh orderings."""
        H4 = hx.generate_recursive(4)
        W_natural = hx.generate_walsh(4, hx.Ordering.NATURAL)
        W_sequency = hx.generate_walsh(4, hx.Ordering.SEQUENCY)
        W_dyadic = hx.generate_walsh(4, hx.Ordering.DYADIC)
        
        # Natural ordering should match recursive
        assert np.array_equal(H4, W_natural)
        
        # All orderings should produce valid Hadamard matrices
        assert hx.is_hadamard(W_sequency)
        assert hx.is_hadamard(W_dyadic)
    
    def test_invalid_orders(self):
        """Test invalid order validation."""
        invalid_orders = [0, -1, 3, 5, 6, 7, 9, 10, 15, 33, 100]
        
        for order in invalid_orders:
            with pytest.raises(Exception):  # Should raise some exception
                hx.validate_order(order)


class TestMatrixOperations:
    """Test matrix operations."""
    
    def test_transpose(self):
        """Test matrix transpose."""
        H4 = hx.generate_recursive(4)
        T4 = hx.transpose(H4)
        
        assert H4.shape == T4.shape
        assert np.array_equal(H4, T4.T)
    
    def test_matrix_vector_multiply(self):
        """Test matrix-vector multiplication."""
        H2 = hx.generate_recursive(2)
        v = np.array([1, 2], dtype=np.int32)
        result = hx.multiply(H2, v)
        
        # H(2) * [1, 2] = [1+2, 1-2] = [3, -1]
        expected = np.array([3, -1])
        assert np.array_equal(result, expected)
    
    def test_matrix_matrix_multiply(self):
        """Test matrix-matrix multiplication."""
        H2 = hx.generate_recursive(2)
        H2T = hx.transpose(H2)
        product = hx.multiply_matrices(H2, H2T)
        
        # For Hadamard matrix: H * H^T = n * I
        expected = 2 * np.eye(2)
        assert np.array_equal(product, expected)
    
    def test_dimension_mismatch(self):
        """Test dimension mismatch errors."""
        H2 = hx.generate_recursive(2)
        wrong_size_vector = np.array([1, 2, 3], dtype=np.int32)
        
        with pytest.raises(Exception):
            hx.multiply(H2, wrong_size_vector)


class TestValidation:
    """Test matrix validation functions."""
    
    def test_is_hadamard_valid(self):
        """Test validation of valid Hadamard matrices."""
        sizes = [2, 4, 8, 16, 32]
        
        for size in sizes:
            H = hx.generate_recursive(size)
            assert hx.is_hadamard(H)
            assert hx.is_orthogonal(H)
    
    def test_is_hadamard_invalid(self):
        """Test validation of invalid matrices."""
        # Non-square matrix
        not_square = np.array([[1, -1], [1]])
        assert not hx.is_hadamard(not_square)
        
        # Non-±1 matrix
        not_pm1 = np.array([[1, 2], [-1, 1]])
        assert not hx.is_hadamard(not_pm1)
        
        # Non-orthogonal matrix
        not_orthogonal = np.array([[1, 1], [1, 1]])
        assert not hx.is_hadamard(not_orthogonal)
    
    def test_validate_matrix(self):
        """Test comprehensive matrix validation."""
        H4 = hx.generate_recursive(4)
        issues = hx.validate_matrix(H4)
        assert len(issues) == 0
        
        # Test invalid matrix
        empty = np.array([])
        empty_issues = hx.validate_matrix(empty)
        assert len(empty_issues) > 0
        assert "empty" in empty_issues[0].lower()


class TestTransforms:
    """Test Fast Walsh-Hadamard Transform."""
    
    def test_fwht_basic(self):
        """Test basic FWHT functionality."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        transformed = hx.fwht(data)
        
        assert transformed.shape == data.shape
        
        # Test round-trip property
        round_trip = hx.fwht(transformed)
        assert np.allclose(data, round_trip)
    
    def test_ifwht(self):
        """Test inverse FWHT."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        transformed = hx.fwht(data)
        inverted = hx.ifwht(transformed)
        
        assert np.allclose(data, inverted)
    
    def test_fwht_various_sizes(self):
        """Test FWHT with various sizes."""
        sizes = [2, 4, 8, 16, 32, 64]
        
        for size in sizes:
            test_data = np.arange(1, size + 1, dtype=np.float64)
            
            fwht_result = hx.fwht(test_data)
            round_trip_result = hx.fwht(fwht_result)
            
            assert np.allclose(test_data, round_trip_result)
    
    def test_fwht_invalid_size(self):
        """Test FWHT with invalid size."""
        invalid_data = np.array([1.0, 2.0, 3.0])  # Not power of 2
        
        with pytest.raises(Exception):
            hx.fwht(invalid_data)


class TestSerialization:
    """Test serialization and deserialization."""
    
    def test_serialize_deserialize(self):
        """Test serialization round-trip."""
        H4 = hx.generate_recursive(4)
        
        formats = [hx.Format.COMPACT, hx.Format.VERBOSE, hx.Format.CSV, hx.Format.BINARY]
        
        for fmt in formats:
            serialized = hx.serialize(H4, fmt)
            deserialized = hx.deserialize(serialized, fmt)
            
            assert deserialized.shape == H4.shape
            assert np.array_equal(H4, deserialized)
    
    def test_file_io(self):
        """Test file I/O operations."""
        H4 = hx.generate_recursive(4)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filename = f.name
        
        try:
            # Save to file
            hx.save_to_file(H4, filename, hx.Format.COMPACT)
            
            # Load from file
            loaded = hx.load_from_file(filename, hx.Format.COMPACT)
            assert np.array_equal(H4, loaded)
            
        finally:
            # Clean up
            if os.path.exists(filename):
                os.remove(filename)


class TestProperties:
    """Test matrix properties analysis."""
    
    def test_analyze_properties(self):
        """Test properties analysis."""
        H4 = hx.generate_recursive(4)
        props = hx.analyze_properties(H4)
        
        assert props['size'] == 4.0
        assert props['is_hadamard'] == 1.0
        assert props['is_orthogonal'] == 1.0
        assert props['expected_det_magnitude'] == 16.0
        assert props['expected_condition_number'] == 2.0


class TestUtilities:
    """Test utility functions."""
    
    def test_gray_code_conversion(self):
        """Test Gray code conversion."""
        assert hx.binary_to_gray(0) == 0
        assert hx.binary_to_gray(1) == 1
        assert hx.binary_to_gray(2) == 3
        assert hx.binary_to_gray(3) == 2
        
        # Test round-trip conversion
        for i in range(16):
            gray = hx.binary_to_gray(i)
            back = hx.gray_to_binary(gray)
            assert back == i
    
    def test_print_matrix(self):
        """Test matrix printing."""
        H4 = hx.generate_recursive(4)
        
        # Test different formats
        compact_str = hx.print_matrix(H4, hx.Format.COMPACT)
        verbose_str = hx.print_matrix(H4, hx.Format.VERBOSE)
        
        assert isinstance(compact_str, str)
        assert isinstance(verbose_str, str)
        assert len(compact_str) > 0
        assert len(verbose_str) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_hadamard(self):
        """Test convenience Hadamard creation."""
        H_rec = hx.create_hadamard(4, "recursive")
        H_iter = hx.create_hadamard(4, "iterative")
        
        assert H_rec.shape == (4, 4)
        assert H_iter.shape == (4, 4)
        assert np.array_equal(H_rec, H_iter)
        
        with pytest.raises(ValueError):
            hx.create_hadamard(4, "invalid")
    
    def test_create_walsh(self):
        """Test convenience Walsh creation."""
        W_nat = hx.create_walsh(4, "natural")
        W_seq = hx.create_walsh(4, "sequency")
        W_dyad = hx.create_walsh(4, "dyadic")
        
        assert W_nat.shape == (4, 4)
        assert W_seq.shape == (4, 4)
        assert W_dyad.shape == (4, 4)
        
        with pytest.raises(ValueError):
            hx.create_walsh(4, "invalid")


class TestBenchmarking:
    """Test benchmarking functionality."""
    
    def test_benchmark(self):
        """Test benchmarking function."""
        def test_func():
            H = hx.generate_recursive(16)
            return H
        
        time_us = hx.benchmark(test_func, 10)
        assert time_us >= 0
        assert time_us < 1000000  # Should complete in reasonable time


class TestErrorHandling:
    """Test error handling."""
    
    def test_large_matrix_generation(self):
        """Test generation of larger matrices."""
        sizes = [64, 128, 256]
        
        for size in sizes:
            H = hx.generate_iterative(size)  # Use iterative for large sizes
            assert hx.is_hadamard(H)
    
    def test_stress_testing(self):
        """Test stress scenarios."""
        # Test large transforms
        sizes = [64, 128, 256, 512, 1024]
        
        for size in sizes:
            data = np.random.randn(size).astype(np.float64)
            
            transformed = hx.fwht(data)
            round_trip = hx.ifwht(transformed)
            
            assert np.allclose(data, round_trip, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
