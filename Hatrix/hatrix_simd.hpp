//======================================================================
// hatrix_simd.hpp
//----------------------------------------------------------------------
// High-performance SIMD-optimized implementations for Hadamard matrices.
// Features AVX2/AVX-512 vectorization, cache optimization, and threading.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_SIMD_HPP
#define HATRIX_SIMD_HPP

#include "hadamard_matrix.hpp"
#include <immintrin.h>  // AVX2/AVX-512 intrinsics
#include <thread>
#include <future>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

namespace hatrix {
namespace simd {

//--------------------------------------------------------------------------
// SIMD DETECTION AND CONFIGURATION
//--------------------------------------------------------------------------
struct SIMDCapabilities {
    bool avx2 = false;
    bool avx512f = false;
    bool fma = false;
    int max_threads = 1;
    
    SIMDCapabilities() {
        detect_capabilities();
    }
    
private:
    void detect_capabilities() {
        // Detect CPU features
        #ifdef _MSC_VER
            int cpuInfo[4];
            __cpuid(cpuInfo, 1);
            avx2 = (cpuInfo[2] & (1 << 28)) != 0;
            fma = (cpuInfo[2] & (1 << 12)) != 0;
            
            // Check for AVX-512
            __cpuid(cpuInfo, 7);
            avx512f = (cpuInfo[1] & (1 << 16)) != 0;
        #else
            unsigned eax, ebx, ecx, edx;
            __cpuid(1, eax, ebx, ecx, edx);
            avx2 = (ecx & (1 << 28)) != 0;
            fma = (ecx & (1 << 12)) != 0;
            
            __cpuid(7, eax, ebx, ecx, edx);
            avx512f = (ebx & (1 << 16)) != 0;
        #endif
        
        max_threads = std::thread::hardware_concurrency();
        if (max_threads == 0) max_threads = 1;
    }
};

// Global SIMD capabilities
const SIMDCapabilities g_simd_caps;

//--------------------------------------------------------------------------
// MEMORY ALIGNMENT AND ALLOCATION
//--------------------------------------------------------------------------
template<typename T>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    static constexpr size_type alignment = 64;  // Cache line alignment
    
    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U>;
    };
    
    AlignedAllocator() = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>&) {}
    
    pointer allocate(size_type n) {
        if (n > max_size()) {
            throw std::bad_alloc();
        }
        
        void* ptr = nullptr;
        #ifdef _WIN32
            ptr = _aligned_malloc(n * sizeof(T), alignment);
        #else
            if (posix_memalign(&ptr, alignment, n * sizeof(T)) != 0) {
                ptr = nullptr;
            }
        #endif
        
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<pointer>(ptr);
    }
    
    void deallocate(pointer p, size_type) {
        #ifdef _WIN32
            _aligned_free(p);
        #else
            std::free(p);
        #endif
    }
    
    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
    
    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator&) const { return false; }
};

// Aligned vector type
template<typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

//--------------------------------------------------------------------------
// SIMD-OPTIMIZED FWHT IMPLEMENTATIONS
//--------------------------------------------------------------------------
class FWHTOptimized {
public:
    // AVX2-optimized FWHT
    static aligned_vector<double> fwht_avx2(const aligned_vector<double>& data) {
        aligned_vector<double> result = data;
        size_t n = result.size();
        
        // Ensure data is aligned and padded
        size_t padded_n = align_size(n);
        result.resize(padded_n, 0.0);
        
        // Perform FWHT with AVX2
        for (size_t len = 2; len <= padded_n; len <<= 1) {
            size_t half_len = len >> 1;
            
            for (size_t i = 0; i < padded_n; i += len) {
                for (size_t j = 0; j < half_len; j += 4) {  // Process 4 elements at once
                    if (j + 4 <= half_len) {
                        __m256d u = _mm256_load_pd(&result[i + j]);
                        __m256d v = _mm256_load_pd(&result[i + j + half_len]);
                        
                        __m256d sum = _mm256_add_pd(u, v);
                        __m256d diff = _mm256_sub_pd(u, v);
                        
                        _mm256_store_pd(&result[i + j], sum);
                        _mm256_store_pd(&result[i + j + half_len], diff);
                    } else {
                        // Handle remaining elements
                        for (size_t k = j; k < half_len; ++k) {
                            double u = result[i + k];
                            double v = result[i + k + half_len];
                            result[i + k] = u + v;
                            result[i + k + half_len] = u - v;
                        }
                    }
                }
            }
        }
        
        // Normalize
        __m256d norm = _mm256_set1_pd(1.0 / std::sqrt(n));
        for (size_t i = 0; i < padded_n; i += 4) {
            __m256d data_vec = _mm256_load_pd(&result[i]);
            __m256d normalized = _mm256_mul_pd(data_vec, norm);
            _mm256_store_pd(&result[i], normalized);
        }
        
        // Resize back to original size
        result.resize(n);
        return result;
    }
    
    // AVX-512 optimized FWHT (when available)
    static aligned_vector<double> fwht_avx512(const aligned_vector<double>& data) {
        aligned_vector<double> result = data;
        size_t n = result.size();
        
        if (!g_simd_caps.avx512f) {
            return fwht_avx2(data);  // Fallback to AVX2
        }
        
        size_t padded_n = align_size(n);
        result.resize(padded_n, 0.0);
        
        // Perform FWHT with AVX-512
        for (size_t len = 2; len <= padded_n; len <<= 1) {
            size_t half_len = len >> 1;
            
            for (size_t i = 0; i < padded_n; i += len) {
                for (size_t j = 0; j < half_len; j += 8) {  // Process 8 elements at once
                    if (j + 8 <= half_len) {
                        __m512d u = _mm512_load_pd(&result[i + j]);
                        __m512d v = _mm512_load_pd(&result[i + j + half_len]);
                        
                        __m512d sum = _mm512_add_pd(u, v);
                        __m512d diff = _mm512_sub_pd(u, v);
                        
                        _mm512_store_pd(&result[i + j], sum);
                        _mm512_store_pd(&result[i + j + half_len], diff);
                    } else {
                        // Handle remaining elements with AVX2
                        for (size_t k = j; k < half_len; k += 4) {
                            if (k + 4 <= half_len) {
                                __m256d u = _mm256_load_pd(&result[i + k]);
                                __m256d v = _mm256_load_pd(&result[i + k + half_len]);
                                
                                __m256d sum = _mm256_add_pd(u, v);
                                __m256d diff = _mm256_sub_pd(u, v);
                                
                                _mm256_store_pd(&result[i + k], sum);
                                _mm256_store_pd(&result[i + k + half_len], diff);
                            } else {
                                // Handle remaining elements
                                for (size_t l = k; l < half_len; ++l) {
                                    double u_val = result[i + l];
                                    double v_val = result[i + l + half_len];
                                    result[i + l] = u_val + v_val;
                                    result[i + l + half_len] = u_val - v_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Normalize with AVX-512
        __m512d norm = _mm512_set1_pd(1.0 / std::sqrt(n));
        for (size_t i = 0; i < padded_n; i += 8) {
            __m512d data_vec = _mm512_load_pd(&result[i]);
            __m512d normalized = _mm512_mul_pd(data_vec, norm);
            _mm512_store_pd(&result[i], normalized);
        }
        
        result.resize(n);
        return result;
    }
    
    // Multi-threaded FWHT
    static aligned_vector<double> fwht_parallel(const aligned_vector<double>& data) {
        size_t n = data.size();
        if (n < 1024 || g_simd_caps.max_threads <= 1) {
            return fwht_avx2(data);  // Use single-threaded SIMD for small data
        }
        
        aligned_vector<double> result = data;
        size_t padded_n = align_size(n);
        result.resize(padded_n, 0.0);
        
        // Parallel FWHT implementation
        size_t num_threads = std::min(g_simd_caps.max_threads, static_cast<int>(std::log2(padded_n)));
        
        for (size_t len = 2; len <= padded_n; len <<= 1) {
            size_t half_len = len >> 1;
            size_t block_size = (padded_n / len) / num_threads;
            if (block_size == 0) block_size = 1;
            
            std::vector<std::future<void>> futures;
            
            for (int t = 0; t < num_threads; ++t) {
                size_t start_block = t * block_size;
                size_t end_block = std::min(start_block + block_size, padded_n / len);
                
                futures.emplace_back(std::async(std::launch::async, [&, start_block, end_block, len, half_len]() {
                    for (size_t block = start_block; block < end_block; ++block) {
                        size_t i = block * len;
                        fwht_block_avx2(result.data() + i, half_len);
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (auto& future : futures) {
                future.wait();
            }
        }
        
        // Normalize
        normalize_parallel(result.data(), padded_n, n);
        
        result.resize(n);
        return result;
    }
    
private:
    static size_t align_size(size_t n) {
        // Align to multiple of 8 for AVX-512 or 4 for AVX2
        size_t alignment = g_simd_caps.avx512f ? 8 : 4;
        return ((n + alignment - 1) / alignment) * alignment;
    }
    
    static void fwht_block_avx2(double* data, size_t half_len) {
        for (size_t j = 0; j < half_len; j += 4) {
            if (j + 4 <= half_len) {
                __m256d u = _mm256_load_pd(data + j);
                __m256d v = _mm256_load_pd(data + j + half_len);
                
                __m256d sum = _mm256_add_pd(u, v);
                __m256d diff = _mm256_sub_pd(u, v);
                
                _mm256_store_pd(data + j, sum);
                _mm256_store_pd(data + j + half_len, diff);
            } else {
                // Handle remaining elements
                for (size_t k = j; k < half_len; ++k) {
                    double u = data[k];
                    double v = data[k + half_len];
                    data[k] = u + v;
                    data[k + half_len] = u - v;
                }
            }
        }
    }
    
    static void normalize_parallel(double* data, size_t padded_n, size_t original_n) {
        double norm = 1.0 / std::sqrt(original_n);
        __m256d norm_vec = _mm256_set1_pd(norm);
        
        size_t num_threads = std::min(g_simd_caps.max_threads, 4);
        size_t block_size = padded_n / num_threads;
        
        std::vector<std::future<void>> futures;
        
        for (int t = 0; t < num_threads; ++t) {
            size_t start = t * block_size;
            size_t end = (t == num_threads - 1) ? padded_n : (t + 1) * block_size;
            
            futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
                for (size_t i = start; i < end; i += 4) {
                    if (i + 4 <= end) {
                        __m256d data_vec = _mm256_load_pd(data + i);
                        __m256d normalized = _mm256_mul_pd(data_vec, norm_vec);
                        _mm256_store_pd(data + i, normalized);
                    }
                }
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
    }
};

//--------------------------------------------------------------------------
// SIMD-OPTIMIZED MATRIX OPERATIONS
//--------------------------------------------------------------------------
class MatrixOptimized {
public:
    // Cache-optimized matrix generation with SIMD
    static aligned_vector<int> generate_hadamard_blocked(int n) {
        if (n == 1) {
            aligned_vector<int> result(1);
            result[0] = 1;
            return result;
        }
        
        aligned_vector<int> result(n * n);
        
        // Block size for cache optimization
        constexpr int block_size = 64;  // 64x64 blocks fit in L1 cache
        
        // Use recursive approach but with blocked memory access
        generate_hadamard_blocked_recursive(result.data(), n, 0, 0, block_size);
        
        return result;
    }
    
    // SIMD-optimized matrix transpose
    static aligned_vector<int> transpose_simd(const aligned_vector<int>& matrix, int n) {
        aligned_vector<int> result(n * n);
        
        // Block size for cache optimization
        constexpr int block_size = 32;
        
        for (int i = 0; i < n; i += block_size) {
            for (int j = 0; j < n; j += block_size) {
                transpose_block(matrix.data(), result.data(), n, i, j, block_size);
            }
        }
        
        return result;
    }
    
    // SIMD-optimized matrix-vector multiplication
    static aligned_vector<int> multiply_vector_simd(const aligned_vector<int>& matrix, 
                                                   const aligned_vector<int>& vector, int n) {
        aligned_vector<int> result(n);
        
        // Use AVX2 for 32-bit integers (8 elements at once)
        for (int i = 0; i < n; ++i) {
            __m256i sum = _mm256_setzero_si256();
            const int* row = matrix.data() + i * n;
            
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256i matrix_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(row + j));
                __m256i vector_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(vector.data() + j));
                
                __m256i product = _mm256_mullo_epi32(matrix_vec, vector_vec);
                sum = _mm256_add_epi32(sum, product);
            }
            
            // Handle remaining elements
            int partial_sum = 0;
            for (; j < n; ++j) {
                partial_sum += row[j] * vector[j];
            }
            
            // Extract sum from SIMD register
            int simd_sum[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(simd_sum), sum);
            
            result[i] = partial_sum + simd_sum[0] + simd_sum[1] + simd_sum[2] + simd_sum[3] +
                       simd_sum[4] + simd_sum[5] + simd_sum[6] + simd_sum[7];
        }
        
        return result;
    }
    
private:
    static void generate_hadamard_blocked_recursive(int* matrix, int n, int row_start, int col_start, int block_size) {
        if (n <= block_size) {
            // Generate small block directly
            if (n == 1) {
                matrix[row_start * n + col_start] = 1;
            } else if (n == 2) {
                matrix[row_start * n + col_start] = 1;
                matrix[row_start * n + col_start + 1] = 1;
                matrix[(row_start + 1) * n + col_start] = 1;
                matrix[(row_start + 1) * n + col_start + 1] = -1;
            } else {
                // Use standard recursive for small blocks
                int half = n / 2;
                generate_hadamard_blocked_recursive(matrix, half, row_start, col_start, block_size);
                generate_hadamard_blocked_recursive(matrix, half, row_start, col_start + half, block_size);
                generate_hadamard_blocked_recursive(matrix, half, row_start + half, col_start, block_size);
                generate_hadamard_blocked_recursive(matrix, half, row_start + half, col_start + half, block_size);
                
                // Negate bottom-right block
                for (int i = 0; i < half; ++i) {
                    for (int j = 0; j < half; ++j) {
                        matrix[(row_start + half + i) * n + col_start + half + j] *= -1;
                    }
                }
            }
        } else {
            // Recursive decomposition for large matrices
            int half = n / 2;
            generate_hadamard_blocked_recursive(matrix, half, row_start, col_start, block_size);
            generate_hadamard_blocked_recursive(matrix, half, row_start, col_start + half, block_size);
            generate_hadamard_blocked_recursive(matrix, half, row_start + half, col_start, block_size);
            generate_hadamard_blocked_recursive(matrix, half, row_start + half, col_start + half, block_size);
            
            // Negate bottom-right block
            for (int i = 0; i < half; ++i) {
                for (int j = 0; j < half; ++j) {
                    matrix[(row_start + half + i) * n + col_start + half + j] *= -1;
                }
            }
        }
    }
    
    static void transpose_block(const int* src, int* dst, int n, int row_start, int col_start, int block_size) {
        int actual_block_size = std::min(block_size, n - row_start);
        actual_block_size = std::min(actual_block_size, n - col_start);
        
        for (int i = 0; i < actual_block_size; ++i) {
            for (int j = 0; j < actual_block_size; ++j) {
                dst[(col_start + j) * n + row_start + i] = src[(row_start + i) * n + col_start + j];
            }
        }
    }
};

//--------------------------------------------------------------------------
// PERFORMANCE MONITORING AND PROFILING
//--------------------------------------------------------------------------
class PerformanceProfiler {
public:
    struct ProfileData {
        double generation_time = 0.0;
        double fwht_time = 0.0;
        double transpose_time = 0.0;
        double multiply_time = 0.0;
        size_t memory_usage = 0;
        int num_threads_used = 1;
        std::string simd_type = "none";
    };
    
    static ProfileData profile_operation(std::function<void()> operation, const std::string& name) {
        ProfileData data;
        auto start = std::chrono::high_resolution_clock::now();
        
        operation();
        
        auto end = std::chrono::high_resolution_clock::now();
        data.generation_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        
        return data;
    }
    
    static void print_performance_summary(const ProfileData& data) {
        std::cout << "Performance Summary:\n";
        std::cout << "  Generation time: " << data.generation_time << " ms\n";
        std::cout << "  FWHT time: " << data.fwht_time << " ms\n";
        std::cout << "  Transpose time: " << data.transpose_time << " ms\n";
        std::cout << "  Multiply time: " << data.multiply_time << " ms\n";
        std::cout << "  Memory usage: " << data.memory_usage / (1024 * 1024) << " MB\n";
        std::cout << "  Threads used: " << data.num_threads_used << "\n";
        std::cout << "  SIMD type: " << data.simd_type << "\n";
    }
};

} // namespace simd
} // namespace hatrix

#endif // HATRIX_SIMD_HPP
