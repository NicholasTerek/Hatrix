//======================================================================
// advanced_benchmark.cpp
//----------------------------------------------------------------------
// Advanced benchmarks including GEMM operations, cache analysis,
// and comprehensive performance comparisons inspired by BLAS optimization.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================

#include <benchmark/benchmark.h>
#include "../Hatrix/hadamard_matrix.hpp"
#include "../Hatrix/hatrix_simd.hpp"
#include "../Hatrix/hatrix_parallel.hpp"
#include "../Hatrix/hatrix_gemm.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace adv {

//--------------------------------------------------------------------------
// BENCHMARK HELPERS
//--------------------------------------------------------------------------
template<typename Func>
void BM_Advanced(benchmark::State& state, Func&& func, const std::string& name) {
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e9);
    }
    
    state.SetItemsProcessed(state.iterations());
}

//--------------------------------------------------------------------------
// NAIVE IMPLEMENTATIONS FOR COMPARISON
//--------------------------------------------------------------------------

// Naive matrix multiplication (RCI - Row, Column, Inner)
template<int M, int N, int K>
void naive_gemm_rci(const float* A, const float* B, float* C) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            for (int inner = 0; inner < K; ++inner) {
                C[row * N + col] += A[row * K + inner] * B[inner * N + col];
            }
        }
    }
}

// Cache-aware matrix multiplication (RIC - Row, Inner, Column)
template<int M, int N, int K>
void cache_aware_gemm_ric(const float* A, const float* B, float* C) {
    for (int row = 0; row < M; ++row) {
        for (int inner = 0; inner < K; ++inner) {
            for (int col = 0; col < N; ++col) {
                C[row * N + col] += A[row * K + inner] * B[inner * N + col];
            }
        }
    }
}

// Tiled matrix multiplication
template<int M, int N, int K, int TILE_SIZE>
void tiled_gemm(const float* A, const float* B, float* C) {
    for (int ii = 0; ii < M; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < K; kk += TILE_SIZE) {
                
                int i_end = std::min(ii + TILE_SIZE, M);
                int j_end = std::min(jj + TILE_SIZE, N);
                int k_end = std::min(kk + TILE_SIZE, K);
                
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        for (int j = jj; j < j_end; ++j) {
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// GEMM BENCHMARKS
//--------------------------------------------------------------------------

static void BM_NaiveGEMM_1024(benchmark::State& state) {
    constexpr int N = 1024;
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : A) val = dis(gen);
    for (auto& val : B) val = dis(gen);
    
    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);
        naive_gemm_rci<N, N, N>(A.data(), B.data(), C.data());
        benchmark::DoNotOptimize(C);
    }
    
    state.SetItemsProcessed(state.iterations() * N * N * N * 2); // 2 FLOPs per multiply-add
    state.SetBytesProcessed(state.iterations() * (N * N * 3) * sizeof(float));
}

static void BM_CacheAwareGEMM_1024(benchmark::State& state) {
    constexpr int N = 1024;
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : A) val = dis(gen);
    for (auto& val : B) val = dis(gen);
    
    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);
        cache_aware_gemm_ric<N, N, N>(A.data(), B.data(), C.data());
        benchmark::DoNotOptimize(C);
    }
    
    state.SetItemsProcessed(state.iterations() * N * N * N * 2);
    state.SetBytesProcessed(state.iterations() * (N * N * 3) * sizeof(float));
}

static void BM_TiledGEMM_1024(benchmark::State& state) {
    constexpr int N = 1024;
    constexpr int TILE_SIZE = 64;
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : A) val = dis(gen);
    for (auto& val : B) val = dis(gen);
    
    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);
        tiled_gemm<N, N, N, TILE_SIZE>(A.data(), B.data(), C.data());
        benchmark::DoNotOptimize(C);
    }
    
    state.SetItemsProcessed(state.iterations() * N * N * N * 2);
    state.SetBytesProcessed(state.iterations() * (N * N * 3) * sizeof(float));
}

static void BM_AdvancedGEMM_1024(benchmark::State& state) {
    constexpr int N = 1024;
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : A) val = dis(gen);
    for (auto& val : B) val = dis(gen);
    
    for (auto _ : state) {
        auto C = hatrix::gemm::AdvancedGEMM::multiply(A, B, N, N, N, true);
        benchmark::DoNotOptimize(C);
    }
    
    state.SetItemsProcessed(state.iterations() * N * N * N * 2);
    state.SetBytesProcessed(state.iterations() * (N * N * 3) * sizeof(float));
}

//--------------------------------------------------------------------------
// HADAMARD MATRIX BENCHMARKS
//--------------------------------------------------------------------------

static void BM_HadamardGeneration_Naive(benchmark::State& state) {
    int size = state.range(0);
    
    for (auto _ : state) {
        auto matrix = hadamard::generate_recursive(size);
        benchmark::DoNotOptimize(matrix);
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

static void BM_HadamardGeneration_Optimized(benchmark::State& state) {
    int size = state.range(0);
    hatrix::parallel::ParallelHadamardGenerator generator;
    
    for (auto _ : state) {
        auto matrix = generator.generate_simd_parallel(size);
        benchmark::DoNotOptimize(matrix);
    }
    
    state.SetItemsProcessed(state.iterations() * size * size);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

static void BM_HadamardMatrixMultiply_Naive(benchmark::State& state) {
    int size = state.range(0);
    auto H = hadamard::generate_recursive(size);
    
    for (auto _ : state) {
        auto result = hadamard::multiply(H, H);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * size * size * size * 2);
    state.SetBytesProcessed(state.iterations() * size * size * size * sizeof(int) * 3);
}

static void BM_HadamardMatrixMultiply_Optimized(benchmark::State& state) {
    int size = state.range(0);
    hatrix::parallel::ParallelHadamardGenerator generator;
    auto H = generator.generate_simd_parallel(size);
    
    for (auto _ : state) {
        auto result = hatrix::gemm::CacheOptimizedHadamard::multiply_hadamard_optimized(
            std::vector<int>(H.data(), H.data() + H.size()),
            std::vector<int>(H.data(), H.data() + H.size()),
            size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * size * size * size * 2);
    state.SetBytesProcessed(state.iterations() * size * size * size * sizeof(int) * 3);
}

//--------------------------------------------------------------------------
// FWHT BENCHMARKS
//--------------------------------------------------------------------------

static void BM_FWHT_Naive(benchmark::State& state) {
    int size = state.range(0);
    std::vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    for (auto _ : state) {
        auto result = hadamard::fwht(data);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(double) * 2);
}

static void BM_FWHT_AVX2(benchmark::State& state) {
    int size = state.range(0);
    hatrix::simd::aligned_vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    for (auto _ : state) {
        auto result = hatrix::simd::FWHTOptimized::fwht_avx2(data);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(double) * 2);
}

static void BM_FWHT_Parallel(benchmark::State& state) {
    int size = state.range(0);
    hatrix::simd::aligned_vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    for (auto _ : state) {
        auto result = hatrix::simd::FWHTOptimized::fwht_parallel(data);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(double) * 2);
}

//--------------------------------------------------------------------------
// BATCH PROCESSING BENCHMARKS
//--------------------------------------------------------------------------

static void BM_BatchFWHT_Sequential(benchmark::State& state) {
    int batch_size = state.range(0);
    int signal_size = state.range(1);
    
    std::vector<std::vector<double>> signals(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        signals[i].resize(signal_size);
        std::iota(signals[i].begin(), signals[i].end(), i * signal_size + 1.0);
    }
    
    for (auto _ : state) {
        std::vector<std::vector<double>> results(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            results[i] = hadamard::fwht(signals[i]);
        }
        benchmark::DoNotOptimize(results);
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetBytesProcessed(state.iterations() * batch_size * signal_size * sizeof(double) * 2);
}

static void BM_BatchFWHT_Parallel(benchmark::State& state) {
    int batch_size = state.range(0);
    int signal_size = state.range(1);
    
    std::vector<hatrix::simd::aligned_vector<double>> signals(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        signals[i].resize(signal_size);
        std::iota(signals[i].begin(), signals[i].end(), i * signal_size + 1.0);
    }
    
    hatrix::parallel::BatchProcessor processor;
    
    for (auto _ : state) {
        auto results = processor.process_signals_batch(signals);
        benchmark::DoNotOptimize(results);
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetBytesProcessed(state.iterations() * batch_size * signal_size * sizeof(double) * 2);
}

//--------------------------------------------------------------------------
// PERFORMANCE ANALYSIS BENCHMARKS
//--------------------------------------------------------------------------

static void BM_PerformanceAnalysis(benchmark::State& state) {
    int size = state.range(0);
    
    // Generate test data
    std::vector<float> A(size * size);
    std::vector<float> B(size * size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : A) val = dis(gen);
    for (auto& val : B) val = dis(gen);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto C = hatrix::gemm::AdvancedGEMM::multiply(A, B, size, size, size, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        auto metrics = hatrix::gemm::AdvancedGEMM::calculate_metrics(size, size, size, time_ms);
        
        // Set custom counters
        state.counters["GFLOPS"] = metrics.flops_per_second / 1e9;
        state.counters["Memory_GBps"] = metrics.memory_bandwidth_gbps;
        state.counters["Efficiency_%"] = metrics.efficiency_percent;
        
        benchmark::DoNotOptimize(C);
    }
    
    state.SetItemsProcessed(state.iterations() * size * size * size * 2);
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(float) * 3);
}

//--------------------------------------------------------------------------
// BENCHMARK REGISTRATION
//--------------------------------------------------------------------------

// GEMM benchmarks
BENCHMARK(BM_NaiveGEMM_1024)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_CacheAwareGEMM_1024)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_TiledGEMM_1024)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_AdvancedGEMM_1024)->Unit(benchmark::kMillisecond)->UseRealTime();

// Hadamard matrix benchmarks
BENCHMARK(BM_HadamardGeneration_Naive)->RangeMultiplier(2)->Range(64, 2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_HadamardGeneration_Optimized)->RangeMultiplier(2)->Range(64, 2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_HadamardMatrixMultiply_Naive)->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_HadamardMatrixMultiply_Optimized)->RangeMultiplier(2)->Range(64, 1024)->Unit(benchmark::kMicrosecond);

// FWHT benchmarks
BENCHMARK(BM_FWHT_Naive)->RangeMultiplier(2)->Range(64, 8192)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FWHT_AVX2)->RangeMultiplier(2)->Range(64, 8192)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FWHT_Parallel)->RangeMultiplier(2)->Range(64, 8192)->Unit(benchmark::kMicrosecond);

// Batch processing benchmarks
BENCHMARK(BM_BatchFWHT_Sequential)
    ->Args({100, 256})
    ->Args({100, 1024})
    ->Args({1000, 256})
    ->Args({1000, 1024})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_BatchFWHT_Parallel)
    ->Args({100, 256})
    ->Args({100, 1024})
    ->Args({1000, 256})
    ->Args({1000, 1024})
    ->Unit(benchmark::kMillisecond);

// Performance analysis benchmarks
BENCHMARK(BM_PerformanceAnalysis)
    ->RangeMultiplier(2)
    ->Range(256, 2048)
    ->Unit(benchmark::kMillisecond);

} // namespace adv

//--------------------------------------------------------------------------
// MAIN FUNCTION
//--------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Print system information
    std::cout << "Hatrix Advanced Performance Benchmark Suite" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "SIMD Capabilities:" << std::endl;
    std::cout << "  AVX2: " << (hatrix::simd::g_simd_caps.avx2 ? "Yes" : "No") << std::endl;
    std::cout << "  AVX-512: " << (hatrix::simd::g_simd_caps.avx512f ? "Yes" : "No") << std::endl;
    std::cout << "  FMA: " << (hatrix::simd::g_simd_caps.fma ? "Yes" : "No") << std::endl;
    std::cout << "  Max Threads: " << hatrix::simd::g_simd_caps.max_threads << std::endl;
    std::cout << std::endl;
    
    // Run performance analysis for different matrix sizes
    std::cout << "Performance Analysis:" << std::endl;
    std::cout << "Size\tTime(ms)\tGFLOPS\tMemory(GB/s)\tEfficiency(%)" << std::endl;
    std::cout << "----\t--------\t------\t------------\t--------------" << std::endl;
    
    for (int size : {256, 512, 1024, 2048}) {
        double time_ms = hatrix::gemm::AdvancedGEMM::benchmark_gemm(size, size, size);
        auto metrics = hatrix::gemm::AdvancedGEMM::calculate_metrics(size, size, size, time_ms);
        
        std::cout << std::fixed << std::setprecision(2)
                  << size << "\t" << time_ms << "\t\t"
                  << metrics.flops_per_second / 1e9 << "\t\t"
                  << metrics.memory_bandwidth_gbps << "\t\t"
                  << metrics.efficiency_percent << std::endl;
    }
    std::cout << std::endl;
    
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
