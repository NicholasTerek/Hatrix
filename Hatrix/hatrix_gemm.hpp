//======================================================================
// hatrix_gemm.hpp
//----------------------------------------------------------------------
// Advanced GEMM (General Matrix Multiply) implementations with
// cache-aware tiling, register blocking, and SIMD optimization.
// Inspired by high-performance BLAS implementations.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_GEMM_HPP
#define HATRIX_GEMM_HPP

#include "hatrix_simd.hpp"
#include "hatrix_parallel.hpp"
#include <immintrin.h>
#include <thread>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

namespace hatrix {
namespace gemm {

//--------------------------------------------------------------------------
// PERFORMANCE METRICS STRUCTURE
//--------------------------------------------------------------------------
struct PerformanceMetrics {
    double time_ms;
    double flops;
    double flops_per_second;
    double memory_bandwidth_gbps;
    double efficiency_percent;
};

//--------------------------------------------------------------------------
// GEMM CONFIGURATION AND CONSTANTS
//--------------------------------------------------------------------------
struct GEMMConfig {
    // Cache sizes (bytes) - these should be detected at runtime
    int l1_cache_size = 32768;      // 32KB L1 cache
    int l2_cache_size = 262144;     // 256KB L2 cache
    int l3_cache_size = 8388608;    // 8MB L3 cache
    
    // Tiling parameters
    int mc = 256;  // Row blocking size
    int kc = 256;  // Inner dimension blocking size
    int nc = 256;  // Column blocking size
    
    // Register blocking parameters
    int mr = 8;    // Row micro-kernel size
    int nr = 8;    // Column micro-kernel size
    
    // Threading
    int num_threads = std::thread::hardware_concurrency();
    
    void auto_tune(int m, int n, int k) {
        // Auto-tune based on matrix dimensions
        if (m * n * k < l1_cache_size / 4) {
            // Small matrices - use L1 cache
            mc = std::min(m, 64);
            nc = std::min(n, 64);
            kc = std::min(k, 64);
        } else if (m * n * k < l2_cache_size / 4) {
            // Medium matrices - use L2 cache
            mc = std::min(m, 128);
            nc = std::min(n, 128);
            kc = std::min(k, 128);
        } else {
            // Large matrices - use L3 cache
            mc = std::min(m, 256);
            nc = std::min(n, 256);
            kc = std::min(k, 256);
        }
        
        // Ensure dimensions are multiples of micro-kernel sizes
        mc = (mc / mr) * mr;
        nc = (nc / nr) * nr;
        kc = ((kc + 7) / 8) * 8;  // Align to 8 for SIMD
    }
};

//--------------------------------------------------------------------------
// MICRO-KERNELS FOR REGISTER BLOCKING
//--------------------------------------------------------------------------
class MicroKernel {
public:
    // AVX2 micro-kernel for 8x8 blocks
    static void gemm_8x8_avx2(const float* A, const float* B, float* C, 
                              int ldA, int ldB, int ldC, int k) {
        // Load C matrix into registers
        __m256 c00 = _mm256_load_ps(&C[0 * ldC + 0]);
        __m256 c01 = _mm256_load_ps(&C[0 * ldC + 8]);
        __m256 c10 = _mm256_load_ps(&C[8 * ldC + 0]);
        __m256 c11 = _mm256_load_ps(&C[8 * ldC + 8]);
        
        // Main computation loop
        for (int p = 0; p < k; ++p) {
            // Load A column (broadcast to all elements)
            __m256 a0 = _mm256_broadcast_ss(&A[0 * ldA + p]);
            __m256 a1 = _mm256_broadcast_ss(&A[8 * ldA + p]);
            
            // Load B row
            __m256 b0 = _mm256_load_ps(&B[p * ldB + 0]);
            __m256 b1 = _mm256_load_ps(&B[p * ldB + 8]);
            
            // Fused multiply-add
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
        }
        
        // Store results back to C
        _mm256_store_ps(&C[0 * ldC + 0], c00);
        _mm256_store_ps(&C[0 * ldC + 8], c01);
        _mm256_store_ps(&C[8 * ldC + 0], c10);
        _mm256_store_ps(&C[8 * ldC + 8], c11);
    }
    
    // AVX-512 micro-kernel for 16x16 blocks (when available)
    static void gemm_16x16_avx512(const float* A, const float* B, float* C, 
                                  int ldA, int ldB, int ldC, int k) {
        if (!hatrix::simd::g_simd_caps.avx512f) {
            // Fallback to AVX2
            for (int i = 0; i < 16; i += 8) {
                for (int j = 0; j < 16; j += 8) {
                    gemm_8x8_avx2(&A[i * ldA], &B[0 * ldB + j], &C[i * ldC + j], 
                                  ldA, ldB, ldC, k);
                }
            }
            return;
        }
        
        // Load C matrix into registers (16x16 block)
        __m512 c[16];
        for (int i = 0; i < 16; ++i) {
            c[i] = _mm512_load_ps(&C[i * ldC]);
        }
        
        // Main computation loop
        for (int p = 0; p < k; ++p) {
            // Load A column
            __m512 a = _mm512_set1_ps(A[p]);
            
            // Load B row
            __m512 b = _mm512_load_ps(&B[p * ldB]);
            
            // Fused multiply-add
            for (int i = 0; i < 16; ++i) {
                c[i] = _mm512_fmadd_ps(a, b, c[i]);
            }
        }
        
        // Store results back to C
        for (int i = 0; i < 16; ++i) {
            _mm512_store_ps(&C[i * ldC], c[i]);
        }
    }
    
    // Generic micro-kernel that chooses the best available implementation
    static void gemm_micro_kernel(const float* A, const float* B, float* C, 
                                  int ldA, int ldB, int ldC, int m, int n, int k) {
        if (m == 8 && n == 8) {
            gemm_8x8_avx2(A, B, C, ldA, ldB, ldC, k);
        } else if (m == 16 && n == 16 && hatrix::simd::g_simd_caps.avx512f) {
            gemm_16x16_avx512(A, B, C, ldA, ldB, ldC, k);
        } else {
            // Generic implementation for other sizes
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; ++p) {
                        sum += A[i * ldA + p] * B[p * ldB + j];
                    }
                    C[i * ldC + j] += sum;
                }
            }
        }
    }
};

//--------------------------------------------------------------------------
// CACHE-AWARE TILING IMPLEMENTATIONS
//--------------------------------------------------------------------------
class TiledGEMM {
public:
    explicit TiledGEMM(const GEMMConfig& config = GEMMConfig()) : config_(config) {}
    
    // Main GEMM function: C = alpha * A * B + beta * C
    void gemm(int m, int n, int k, float alpha, const float* A, int ldA,
              const float* B, int ldB, float beta, float* C, int ldC) {
        
        // Auto-tune configuration
        GEMMConfig tuned_config = config_;
        tuned_config.auto_tune(m, n, k);
        
        // Handle beta scaling
        if (beta != 1.0f) {
            scale_matrix(C, m, n, ldC, beta);
        }
        
        // Main tiled computation
        for (int jc = 0; jc < n; jc += tuned_config.nc) {
            int nc = std::min(tuned_config.nc, n - jc);
            
            for (int pc = 0; pc < k; pc += tuned_config.kc) {
                int kc = std::min(tuned_config.kc, k - pc);
                
                // Pack B panel
                auto B_packed = pack_B_panel(&B[pc * ldB + jc], ldB, kc, nc);
                
                for (int ic = 0; ic < m; ic += tuned_config.mc) {
                    int mc = std::min(tuned_config.mc, m - ic);
                    
                    // Pack A panel
                    auto A_packed = pack_A_panel(&A[ic * ldA + pc], ldA, mc, kc);
                    
                    // Compute micro-kernel
                    inner_kernel(A_packed.get(), B_packed.get(), 
                               &C[ic * ldC + jc], ldC, mc, nc, kc, alpha);
                }
            }
        }
    }
    
    // Multi-threaded GEMM
    void gemm_parallel(int m, int n, int k, float alpha, const float* A, int ldA,
                       const float* B, int ldB, float beta, float* C, int ldC) {
        
        // Auto-tune configuration
        GEMMConfig tuned_config = config_;
        tuned_config.auto_tune(m, n, k);
        
        // Handle beta scaling
        if (beta != 1.0f) {
            scale_matrix(C, m, n, ldC, beta);
        }
        
        // Create thread pool
        hatrix::parallel::ThreadPool pool(tuned_config.num_threads);
        std::vector<std::future<void>> futures;
        
        // Parallel computation over column blocks
        for (int jc = 0; jc < n; jc += tuned_config.nc) {
            int nc = std::min(tuned_config.nc, n - jc);
            
            futures.emplace_back(pool.enqueue([&, jc, nc, m, n, k, alpha, ldA, ldB, ldC]() {
                // Each thread handles one column block
                for (int pc = 0; pc < k; pc += tuned_config.kc) {
                    int kc = std::min(tuned_config.kc, k - pc);
                    
                    // Pack B panel
                    auto B_packed = pack_B_panel(&B[pc * ldB + jc], ldB, kc, nc);
                    
                    for (int ic = 0; ic < m; ic += tuned_config.mc) {
                        int mc = std::min(tuned_config.mc, m - ic);
                        
                        // Pack A panel
                        auto A_packed = pack_A_panel(&A[ic * ldA + pc], ldA, mc, kc);
                        
                        // Compute micro-kernel
                        inner_kernel(A_packed.get(), B_packed.get(), 
                                   &C[ic * ldC + jc], ldC, mc, nc, kc, alpha);
                    }
                }
            }));
        }
        
        // Wait for all threads to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
    
private:
    GEMMConfig config_;
    
    // Pack A panel for cache efficiency
    std::unique_ptr<float[]> pack_A_panel(const float* A, int ldA, int mc, int kc) {
        auto packed = std::make_unique<float[]>(mc * kc);
        
        for (int i = 0; i < mc; ++i) {
            for (int k = 0; k < kc; ++k) {
                packed[i * kc + k] = A[i * ldA + k];
            }
        }
        
        return packed;
    }
    
    // Pack B panel for cache efficiency
    std::unique_ptr<float[]> pack_B_panel(const float* B, int ldB, int kc, int nc) {
        auto packed = std::make_unique<float[]>(kc * nc);
        
        for (int k = 0; k < kc; ++k) {
            for (int j = 0; j < nc; ++j) {
                packed[k * nc + j] = B[k * ldB + j];
            }
        }
        
        return packed;
    }
    
    // Inner kernel computation
    void inner_kernel(const float* A_packed, const float* B_packed, float* C,
                      int ldC, int mc, int nc, int kc, float alpha) {
        
        for (int jr = 0; jr < nc; jr += config_.nr) {
            int nr = std::min(config_.nr, nc - jr);
            
            for (int ir = 0; ir < mc; ir += config_.mr) {
                int mr = std::min(config_.mr, mc - ir);
                
                // Apply micro-kernel
                MicroKernel::gemm_micro_kernel(&A_packed[ir * kc], 
                                             &B_packed[0 * nc + jr],
                                             &C[ir * ldC + jr], 
                                             kc, nc, ldC, mr, nr, kc);
                
                // Scale by alpha
                if (alpha != 1.0f) {
                    for (int i = 0; i < mr; ++i) {
                        for (int j = 0; j < nr; ++j) {
                            C[(ir + i) * ldC + (jr + j)] *= alpha;
                        }
                    }
                }
            }
        }
    }
    
    // Scale matrix by scalar
    void scale_matrix(float* C, int m, int n, int ldC, float beta) {
        if (beta == 0.0f) {
            // Zero out matrix
            for (int i = 0; i < m; ++i) {
                std::fill(&C[i * ldC], &C[i * ldC + n], 0.0f);
            }
        } else if (beta != 1.0f) {
            // Scale matrix
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    C[i * ldC + j] *= beta;
                }
            }
        }
    }
};

//--------------------------------------------------------------------------
// HIGH-LEVEL GEMM INTERFACE
//--------------------------------------------------------------------------
class AdvancedGEMM {
public:
    // Simple interface for matrix multiplication
    static std::vector<float> multiply(const std::vector<float>& A, const std::vector<float>& B,
                                      int m, int n, int k, bool use_parallel = true) {
        
        std::vector<float> C(m * n, 0.0f);
        
        TiledGEMM gemm;
        
        if (use_parallel) {
            gemm.gemm_parallel(m, n, k, 1.0f, A.data(), k, 
                              B.data(), n, 0.0f, C.data(), n);
        } else {
            gemm.gemm(m, n, k, 1.0f, A.data(), k, 
                     B.data(), n, 0.0f, C.data(), n);
        }
        
        return C;
    }
    
    // Performance benchmark
    static double benchmark_gemm(int m, int n, int k, int iterations = 10) {
        // Generate random matrices
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        std::vector<float> A(m * k);
        std::vector<float> B(k * n);
        
        for (auto& val : A) val = dis(gen);
        for (auto& val : B) val = dis(gen);
        
        // Warmup
        auto C = multiply(A, B, m, n, k, true);
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            C = multiply(A, B, m, n, k, true);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        return total_time / iterations;
    }
    
    // Calculate theoretical performance metrics
    static PerformanceMetrics calculate_metrics(int m, int n, int k, double time_ms) {
        PerformanceMetrics metrics;
        metrics.time_ms = time_ms;
        metrics.flops = 2.0 * m * n * k;  // 2 FLOPs per multiply-add
        metrics.flops_per_second = metrics.flops / (time_ms / 1000.0);
        metrics.memory_bandwidth_gbps = (m * k + k * n + m * n) * sizeof(float) / (time_ms / 1000.0) / 1e9;
        
        // Calculate efficiency based on theoretical peak
        double peak_flops = hatrix::simd::g_simd_caps.max_threads * 3.4e9 * 32;  // Assuming 32 FLOPs/cycle
        metrics.efficiency_percent = (metrics.flops_per_second / peak_flops) * 100.0;
        
        return metrics;
    }
};

//--------------------------------------------------------------------------
// CACHE-OPTIMIZED HADAMARD MATRIX OPERATIONS
//--------------------------------------------------------------------------
class CacheOptimizedHadamard {
public:
    // Cache-optimized Hadamard matrix multiplication
    static std::vector<int> multiply_hadamard_optimized(const std::vector<int>& A, 
                                                        const std::vector<int>& B,
                                                        int n) {
        
        std::vector<int> C(n * n, 0);
        
        // Use tiled approach for cache efficiency
        constexpr int tile_size = 64;  // 64x64 tiles
        
        for (int ii = 0; ii < n; ii += tile_size) {
            for (int jj = 0; jj < n; jj += tile_size) {
                for (int kk = 0; kk < n; kk += tile_size) {
                    
                    int i_end = std::min(ii + tile_size, n);
                    int j_end = std::min(jj + tile_size, n);
                    int k_end = std::min(kk + tile_size, n);
                    
                    // Compute tile
                    for (int i = ii; i < i_end; ++i) {
                        for (int j = jj; j < j_end; ++j) {
                            int sum = 0;
                            for (int k = kk; k < k_end; ++k) {
                                sum += A[i * n + k] * B[k * n + j];
                            }
                            C[i * n + j] += sum;
                        }
                    }
                }
            }
        }
        
        return C;
    }
    
    // SIMD-optimized Hadamard matrix-vector multiplication
    static std::vector<int> multiply_vector_simd(const std::vector<int>& matrix,
                                                 const std::vector<int>& vector,
                                                 int n) {
        
        std::vector<int> result(n);
        
        // Use AVX2 for 32-bit integers
        for (int i = 0; i < n; ++i) {
            __m256i sum = _mm256_setzero_si256();
            const int* row = &matrix[i * n];
            
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256i mat_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j));
                __m256i vec_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&vector[j]));
                
                __m256i product = _mm256_mullo_epi32(mat_vec, vec_vec);
                sum = _mm256_add_epi32(sum, product);
            }
            
            // Handle remaining elements
            int partial_sum = 0;
            for (; j < n; ++j) {
                partial_sum += row[j] * vector[j];
            }
            
            // Extract sum from SIMD register
            int simd_sum[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(simd_sum), sum);
            
            result[i] = partial_sum + simd_sum[0] + simd_sum[1] + simd_sum[2] + simd_sum[3] +
                       simd_sum[4] + simd_sum[5] + simd_sum[6] + simd_sum[7];
        }
        
        return result;
    }
};

} // namespace gemm
} // namespace hatrix

#endif // HATRIX_GEMM_HPP
