//======================================================================
// test_performance_gtest.cpp
//----------------------------------------------------------------------
// Comprehensive tests for high-performance features including SIMD,
// multi-threading, cache optimization, and advanced GEMM operations.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================

#include <gtest/gtest.h>
#include "../Hatrix/hadamard_matrix.hpp"
#include "../Hatrix/hatrix_simd.hpp"
#include "../Hatrix/hatrix_parallel.hpp"
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>

namespace {

//--------------------------------------------------------------------------
// SIMD AND VECTORIZATION TESTS
//--------------------------------------------------------------------------
class SIMDTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure we have test data
        test_sizes_ = {8, 16, 32, 64, 128, 256, 512, 1024};
        
        // Generate test signals
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-10.0, 10.0);
        
        for (int size : test_sizes_) {
            hatrix::simd::aligned_vector<double> signal(size);
            for (int i = 0; i < size; ++i) {
                signal[i] = dis(gen);
            }
            test_signals_[size] = std::move(signal);
        }
    }
    
    std::vector<int> test_sizes_;
    std::map<int, hatrix::simd::aligned_vector<double>> test_signals_;
};

TEST_F(SIMDTest, SIMDCapabilitiesDetection) {
    // Test that SIMD capabilities are properly detected
    EXPECT_GT(hatrix::simd::g_simd_caps.max_threads, 0);
    
    // Print capabilities for debugging
    std::cout << "SIMD Capabilities:" << std::endl;
    std::cout << "  AVX2: " << (hatrix::simd::g_simd_caps.avx2 ? "Yes" : "No") << std::endl;
    std::cout << "  AVX-512: " << (hatrix::simd::g_simd_caps.avx512f ? "Yes" : "No") << std::endl;
    std::cout << "  FMA: " << (hatrix::simd::g_simd_caps.fma ? "Yes" : "No") << std::endl;
    std::cout << "  Max Threads: " << hatrix::simd::g_simd_caps.max_threads << std::endl;
}

TEST_F(SIMDTest, AlignedMemoryAllocation) {
    // Test that aligned allocation works correctly
    hatrix::simd::aligned_vector<double> vec(1024);
    
    // Check alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(vec.data());
    EXPECT_EQ(addr % 64, 0) << "Memory not properly aligned";
    
    // Test basic operations
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<double>(i);
    }
    
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec[i], static_cast<double>(i));
    }
}

TEST_F(SIMDTest, FWHT_AVX2_Correctness) {
    for (int size : {8, 16, 32, 64, 128, 256}) {
        if (test_signals_.find(size) == test_signals_.end()) continue;
        
        const auto& signal = test_signals_[size];
        auto result_avx2 = hatrix::simd::FWHTOptimized::fwht_avx2(signal);
        
        // Compare with reference implementation
        std::vector<double> reference_signal(signal.begin(), signal.end());
        auto reference_result = hadamard::fwht(reference_signal);
        
        // Check correctness
        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(result_avx2[i], reference_result[i], 1e-10) 
                << "AVX2 FWHT mismatch at index " << i << " for size " << size;
        }
    }
}

TEST_F(SIMDTest, FWHT_AVX512_Correctness) {
    for (int size : {8, 16, 32, 64, 128, 256}) {
        if (test_signals_.find(size) == test_signals_.end()) continue;
        
        const auto& signal = test_signals_[size];
        auto result_avx512 = hatrix::simd::FWHTOptimized::fwht_avx512(signal);
        
        // Compare with reference implementation
        std::vector<double> reference_signal(signal.begin(), signal.end());
        auto reference_result = hadamard::fwht(reference_signal);
        
        // Check correctness
        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(result_avx512[i], reference_result[i], 1e-10) 
                << "AVX-512 FWHT mismatch at index " << i << " for size " << size;
        }
    }
}

TEST_F(SIMDTest, FWHT_Parallel_Correctness) {
    for (int size : {64, 128, 256, 512, 1024}) {
        if (test_signals_.find(size) == test_signals_.end()) continue;
        
        const auto& signal = test_signals_[size];
        auto result_parallel = hatrix::simd::FWHTOptimized::fwht_parallel(signal);
        
        // Compare with reference implementation
        std::vector<double> reference_signal(signal.begin(), signal.end());
        auto reference_result = hadamard::fwht(reference_signal);
        
        // Check correctness
        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(result_parallel[i], reference_result[i], 1e-10) 
                << "Parallel FWHT mismatch at index " << i << " for size " << size;
        }
    }
}

//--------------------------------------------------------------------------
// PARALLEL PROCESSING TESTS
//--------------------------------------------------------------------------
class ParallelTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_sizes_ = {32, 64, 128, 256, 512, 1024};
    }
    
    std::vector<int> test_sizes_;
};

TEST_F(ParallelTest, ThreadPoolBasicFunctionality) {
    hatrix::parallel::ThreadPool pool(4);
    
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    
    // Submit multiple tasks
    for (int i = 0; i < 10; ++i) {
        futures.emplace_back(pool.enqueue([&counter, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.fetch_add(i);
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    EXPECT_EQ(counter.load(), 45); // Sum of 0+1+2+...+9
}

TEST_F(ParallelTest, ParallelMatrixGeneration) {
    for (int size : test_sizes_) {
        hatrix::parallel::ParallelHadamardGenerator generator;
        
        // Test blocked generation
        auto blocked_result = generator.generate_blocked(size);
        EXPECT_EQ(blocked_result.rows(), size);
        EXPECT_EQ(blocked_result.cols(), size);
        
        // Test parallel generation
        auto parallel_result = generator.generate_parallel(size);
        EXPECT_EQ(parallel_result.rows(), size);
        EXPECT_EQ(parallel_result.cols(), size);
        
        // Test SIMD parallel generation
        auto simd_result = generator.generate_simd_parallel(size);
        EXPECT_EQ(simd_result.rows(), size);
        EXPECT_EQ(simd_result.cols(), size);
        
        // Verify all methods produce valid Hadamard matrices
        EXPECT_TRUE(hatrix::parallel::validate_matrix_fast(blocked_result));
        EXPECT_TRUE(hatrix::parallel::validate_matrix_fast(parallel_result));
        EXPECT_TRUE(hatrix::parallel::validate_matrix_fast(simd_result));
    }
}

TEST_F(ParallelTest, ParallelMatrixOperations) {
    for (int size : {32, 64, 128, 256}) {
        hatrix::parallel::ParallelHadamardGenerator generator;
        hatrix::parallel::ParallelMatrixOps ops;
        
        auto matrix = generator.generate_blocked(size);
        
        // Test parallel transpose
        auto transposed = ops.transpose_parallel(matrix);
        EXPECT_EQ(transposed.rows(), size);
        EXPECT_EQ(transposed.cols(), size);
        
        // Verify transpose correctness
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                EXPECT_EQ(transposed(i, j), matrix(j, i)) 
                    << "Transpose mismatch at (" << i << ", " << j << ")";
            }
        }
        
        // Test parallel matrix-vector multiplication
        std::vector<int> vector(size);
        std::iota(vector.begin(), vector.end(), 1);
        
        auto result = ops.multiply_vector_parallel(matrix, vector);
        EXPECT_EQ(result.size(), size);
        
        // Verify result is not all zeros
        int sum = std::accumulate(result.begin(), result.end(), 0);
        EXPECT_NE(sum, 0) << "Matrix-vector multiplication result is all zeros";
    }
}

TEST_F(ParallelTest, BatchProcessing) {
    hatrix::parallel::BatchProcessor processor;
    
    // Generate test signals
    std::vector<hatrix::simd::aligned_vector<double>> signals(10);
    for (int i = 0; i < 10; ++i) {
        signals[i].resize(64);
        std::iota(signals[i].begin(), signals[i].end(), i * 64.0);
    }
    
    // Process batch
    auto results = processor.process_signals_batch(signals);
    
    EXPECT_EQ(results.size(), 10);
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(results[i].size(), 64);
        
        // Verify round-trip accuracy
        auto reconstructed = hatrix::simd::FWHTOptimized::fwht_parallel(results[i]);
        
        for (size_t j = 0; j < 64; ++j) {
            EXPECT_NEAR(signals[i][j], reconstructed[j], 1e-10) 
                << "Batch processing round-trip error at signal " << i << ", index " << j;
        }
    }
}

//--------------------------------------------------------------------------
// CACHE OPTIMIZATION TESTS
//--------------------------------------------------------------------------
class CacheOptimizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_sizes_ = {32, 64, 128, 256, 512};
    }
    
    std::vector<int> test_sizes_;
};

TEST_F(CacheOptimizationTest, CacheOptimizedMatrixStorage) {
    for (int size : test_sizes_) {
        hatrix::parallel::CacheOptimizedMatrix<int> matrix(size, size);
        
        EXPECT_EQ(matrix.rows(), size);
        EXPECT_EQ(matrix.cols(), size);
        EXPECT_EQ(matrix.size(), size * size);
        
        // Test basic operations
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix(i, j) = i * size + j;
            }
        }
        
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                EXPECT_EQ(matrix(i, j), i * size + j) 
                    << "Matrix element mismatch at (" << i << ", " << j << ")";
            }
        }
        
        // Test row pointer access
        for (int i = 0; i < size; ++i) {
            const int* row_ptr = matrix.row_ptr(i);
            for (int j = 0; j < size; ++j) {
                EXPECT_EQ(row_ptr[j], i * size + j) 
                    << "Row pointer mismatch at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST_F(CacheOptimizationTest, MemoryAlignment) {
    hatrix::parallel::CacheOptimizedMatrix<double> matrix(128, 128);
    
    // Check that data is properly aligned
    uintptr_t addr = reinterpret_cast<uintptr_t>(matrix.row_ptr(0));
    EXPECT_EQ(addr % 64, 0) << "Matrix data not properly aligned for cache optimization";
}

//--------------------------------------------------------------------------
// PERFORMANCE REGRESSION TESTS
//--------------------------------------------------------------------------
class PerformanceRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up performance thresholds (in milliseconds)
        performance_thresholds_ = {
            {64, 1.0},    // 64x64 matrix generation should take < 1ms
            {256, 5.0},   // 256x256 matrix generation should take < 5ms
            {512, 20.0},  // 512x512 matrix generation should take < 20ms
            {1024, 100.0} // 1024x1024 matrix generation should take < 100ms
        };
        
        fwht_thresholds_ = {
            {64, 0.1},     // 64-point FWHT should take < 0.1ms
            {256, 0.5},    // 256-point FWHT should take < 0.5ms
            {1024, 2.0},   // 1024-point FWHT should take < 2ms
            {4096, 10.0}   // 4096-point FWHT should take < 10ms
        };
    }
    
    std::map<int, double> performance_thresholds_;
    std::map<int, double> fwht_thresholds_;
};

TEST_F(PerformanceRegressionTest, MatrixGenerationPerformance) {
    hatrix::parallel::ParallelHadamardGenerator generator;
    
    for (const auto& [size, threshold] : performance_thresholds_) {
        auto start = std::chrono::high_resolution_clock::now();
        auto matrix = generator.generate_simd_parallel(size);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        
        EXPECT_LT(duration, threshold) 
            << "Matrix generation performance regression: " << size << "x" << size 
            << " took " << duration << "ms (threshold: " << threshold << "ms)";
        
        // Verify correctness
        EXPECT_TRUE(hatrix::parallel::validate_matrix_fast(matrix));
    }
}

TEST_F(PerformanceRegressionTest, FWHTPerformance) {
    for (const auto& [size, threshold] : fwht_thresholds_) {
        hatrix::simd::aligned_vector<double> signal(size);
        std::iota(signal.begin(), signal.end(), 1.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = hatrix::simd::FWHTOptimized::fwht_parallel(signal);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        
        EXPECT_LT(duration, threshold) 
            << "FWHT performance regression: size " << size 
            << " took " << duration << "ms (threshold: " << threshold << "ms)";
        
        // Verify correctness
        std::vector<double> reference_signal(signal.begin(), signal.end());
        auto reference_result = hadamard::fwht(reference_signal);
        
        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(result[i], reference_result[i], 1e-10) 
                << "FWHT correctness error at index " << i;
        }
    }
}

TEST_F(PerformanceRegressionTest, BatchProcessingPerformance) {
    hatrix::parallel::BatchProcessor processor;
    
    // Test batch processing performance
    std::vector<hatrix::simd::aligned_vector<double>> signals(100);
    for (int i = 0; i < 100; ++i) {
        signals[i].resize(256);
        std::iota(signals[i].begin(), signals[i].end(), i * 256.0);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = processor.process_signals_batch(signals);
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    EXPECT_LT(duration, 50.0) 
        << "Batch processing performance regression: 100 signals of size 256 "
        << "took " << duration << "ms (threshold: 50ms)";
    
    EXPECT_EQ(results.size(), 100);
    for (const auto& result : results) {
        EXPECT_EQ(result.size(), 256);
    }
}

//--------------------------------------------------------------------------
// ACCURACY TESTS
//--------------------------------------------------------------------------
class AccuracyTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_sizes_ = {8, 16, 32, 64, 128, 256, 512, 1024};
    }
    
    std::vector<int> test_sizes_;
};

TEST_F(AccuracyTest, FWHT_RoundTripAccuracy) {
    for (int size : test_sizes_) {
        hatrix::simd::aligned_vector<double> signal(size);
        std::iota(signal.begin(), signal.end(), 1.0);
        
        // Forward transform
        auto transformed = hatrix::simd::FWHTOptimized::fwht_parallel(signal);
        
        // Inverse transform
        auto reconstructed = hatrix::simd::FWHTOptimized::fwht_parallel(transformed);
        
        // Check round-trip accuracy
        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(signal[i], reconstructed[i], 1e-12) 
                << "Round-trip accuracy error at index " << i << " for size " << size;
        }
    }
}

TEST_F(AccuracyTest, MatrixOrthogonality) {
    for (int size : {8, 16, 32, 64}) {
        hatrix::parallel::ParallelHadamardGenerator generator;
        auto matrix = generator.generate_simd_parallel(size);
        
        // Check that rows are orthogonal
        for (int i = 0; i < size; ++i) {
            for (int j = i + 1; j < size; ++j) {
                int dot_product = 0;
                for (int k = 0; k < size; ++k) {
                    dot_product += matrix(i, k) * matrix(j, k);
                }
                EXPECT_EQ(dot_product, 0) 
                    << "Rows " << i << " and " << j << " are not orthogonal for size " << size;
            }
        }
        
        // Check that columns are orthogonal
        for (int i = 0; i < size; ++i) {
            for (int j = i + 1; j < size; ++j) {
                int dot_product = 0;
                for (int k = 0; k < size; ++k) {
                    dot_product += matrix(k, i) * matrix(k, j);
                }
                EXPECT_EQ(dot_product, 0) 
                    << "Columns " << i << " and " << j << " are not orthogonal for size " << size;
            }
        }
    }
}

TEST_F(AccuracyTest, MatrixElements) {
    for (int size : {8, 16, 32, 64, 128, 256}) {
        hatrix::parallel::ParallelHadamardGenerator generator;
        auto matrix = generator.generate_simd_parallel(size);
        
        // Check that all elements are ±1
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                EXPECT_TRUE(matrix(i, j) == 1 || matrix(i, j) == -1) 
                    << "Matrix element at (" << i << ", " << j << ") is not ±1 for size " << size;
            }
        }
    }
}

//--------------------------------------------------------------------------
// STRESS TESTS
//--------------------------------------------------------------------------
class StressTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Only run stress tests in Release mode or when explicitly requested
        #ifndef NDEBUG
            GTEST_SKIP() << "Skipping stress tests in Debug mode";
        #endif
    }
};

TEST_F(StressTest, LargeMatrixGeneration) {
    hatrix::parallel::ParallelHadamardGenerator generator;
    
    // Test large matrix generation
    int large_size = 2048;
    auto start = std::chrono::high_resolution_clock::now();
    auto matrix = generator.generate_simd_parallel(large_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Large matrix generation (" << large_size << "x" << large_size 
              << ") took " << duration << "ms" << std::endl;
    
    EXPECT_EQ(matrix.rows(), large_size);
    EXPECT_EQ(matrix.cols(), large_size);
    EXPECT_TRUE(hatrix::parallel::validate_matrix_fast(matrix));
}

TEST_F(StressTest, LargeFWHT) {
    int large_size = 8192;
    hatrix::simd::aligned_vector<double> signal(large_size);
    std::iota(signal.begin(), signal.end(), 1.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = hatrix::simd::FWHTOptimized::fwht_parallel(signal);
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Large FWHT (size " << large_size << ") took " << duration << "ms" << std::endl;
    
    EXPECT_EQ(result.size(), large_size);
    
    // Verify round-trip accuracy
    auto reconstructed = hatrix::simd::FWHTOptimized::fwht_parallel(result);
    for (size_t i = 0; i < large_size; ++i) {
        EXPECT_NEAR(signal[i], reconstructed[i], 1e-10) 
            << "Large FWHT round-trip error at index " << i;
    }
}

TEST_F(StressTest, MassiveBatchProcessing) {
    hatrix::parallel::BatchProcessor processor;
    
    // Test massive batch processing
    std::vector<hatrix::simd::aligned_vector<double>> signals(1000);
    for (int i = 0; i < 1000; ++i) {
        signals[i].resize(512);
        std::iota(signals[i].begin(), signals[i].end(), i * 512.0);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = processor.process_signals_batch(signals);
    auto end = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Massive batch processing (1000 signals of size 512) took " 
              << duration << "ms" << std::endl;
    
    EXPECT_EQ(results.size(), 1000);
    for (const auto& result : results) {
        EXPECT_EQ(result.size(), 512);
    }
}

} // namespace
