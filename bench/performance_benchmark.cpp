//======================================================================
// performance_benchmark.cpp
//----------------------------------------------------------------------
// High-performance benchmarks comparing optimized vs naive implementations.
// Tests SIMD, multi-threading, and cache optimization benefits.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================

#include <benchmark/benchmark.h>
#include "../Hatrix/hadamard_matrix.hpp"
#include "../Hatrix/hatrix_simd.hpp"
#include "../Hatrix/hatrix_parallel.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

namespace perf {

//--------------------------------------------------------------------------
// BENCHMARK HELPERS
//--------------------------------------------------------------------------
template<typename Func>
void BM_Performance(benchmark::State& state, Func&& func, const std::string& name) {
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
// FWHT PERFORMANCE BENCHMARKS
//--------------------------------------------------------------------------

// Naive FWHT implementation for comparison
std::vector<double> naive_fwht(const std::vector<double>& data) {
    std::vector<double> result = data;
    int n = static_cast<int>(result.size());
    
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                double u = result[i + j];
                double v = result[i + j + len / 2];
                result[i + j] = u + v;
                result[i + j + len / 2] = u - v;
            }
        }
    }
    
    // Normalize
    double norm = 1.0 / std::sqrt(n);
    for (auto& val : result) {
        val *= norm;
    }
    
    return result;
}

static void BM_FWHT_Naive(benchmark::State& state) {
    int size = state.range(0);
    std::vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    for (auto _ : state) {
        auto result = naive_fwht(data);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * sizeof(double) * 2);
}

static void BM_FWHT_Original(benchmark::State& state) {
    int size = state.range(0);
    std::vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    for (auto _ : state) {
        auto result = hadamard::fwht(data);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
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
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * sizeof(double) * 2);
}

static void BM_FWHT_AVX512(benchmark::State& state) {
    int size = state.range(0);
    hatrix::simd::aligned_vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    for (auto _ : state) {
        auto result = hatrix::simd::FWHTOptimized::fwht_avx512(data);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
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
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * sizeof(double) * 2);
}

//--------------------------------------------------------------------------
// MATRIX GENERATION PERFORMANCE BENCHMARKS
//--------------------------------------------------------------------------

static void BM_Generate_Recursive_Naive(benchmark::State& state) {
    int size = state.range(0);
    
    for (auto _ : state) {
        auto result = hadamard::generate_recursive(size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

static void BM_Generate_Iterative_Naive(benchmark::State& state) {
    int size = state.range(0);
    
    for (auto _ : state) {
        auto result = hadamard::generate_iterative(size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

static void BM_Generate_Blocked(benchmark::State& state) {
    int size = state.range(0);
    hatrix::parallel::ParallelHadamardGenerator generator;
    
    for (auto _ : state) {
        auto result = generator.generate_blocked(size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

static void BM_Generate_Parallel(benchmark::State& state) {
    int size = state.range(0);
    hatrix::parallel::ParallelHadamardGenerator generator;
    
    for (auto _ : state) {
        auto result = generator.generate_parallel(size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

static void BM_Generate_SIMD_Parallel(benchmark::State& state) {
    int size = state.range(0);
    hatrix::parallel::ParallelHadamardGenerator generator;
    
    for (auto _ : state) {
        auto result = generator.generate_simd_parallel(size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

//--------------------------------------------------------------------------
// MATRIX OPERATIONS PERFORMANCE BENCHMARKS
//--------------------------------------------------------------------------

static void BM_Transpose_Naive(benchmark::State& state) {
    int size = state.range(0);
    auto H = hadamard::generate_recursive(size);
    
    for (auto _ : state) {
        auto result = hadamard::transpose(H);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int) * 2);
}

static void BM_Transpose_SIMD(benchmark::State& state) {
    int size = state.range(0);
    hatrix::parallel::ParallelMatrixOps ops;
    hatrix::parallel::ParallelHadamardGenerator gen;
    auto matrix = gen.generate_blocked(size);
    
    for (auto _ : state) {
        auto result = ops.transpose_parallel(matrix);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int) * 2);
}

static void BM_MatrixVector_Naive(benchmark::State& state) {
    int size = state.range(0);
    auto H = hadamard::generate_recursive(size);
    std::vector<int> v(size);
    std::iota(v.begin(), v.end(), 1);
    
    for (auto _ : state) {
        auto result = hadamard::multiply(H, v);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int) + size * sizeof(int));
}

static void BM_MatrixVector_SIMD(benchmark::State& state) {
    int size = state.range(0);
    hatrix::parallel::ParallelMatrixOps ops;
    hatrix::parallel::ParallelHadamardGenerator gen;
    auto matrix = gen.generate_blocked(size);
    std::vector<int> v(size);
    std::iota(v.begin(), v.end(), 1);
    
    for (auto _ : state) {
        auto result = ops.multiply_vector_parallel(matrix, v);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int) + size * sizeof(int));
}

static void BM_MatrixMatrix_Naive(benchmark::State& state) {
    int size = state.range(0);
    auto H = hadamard::generate_recursive(size);
    
    for (auto _ : state) {
        auto result = hadamard::multiply(H, H);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * size * sizeof(int) * 3);
}

static void BM_MatrixMatrix_Parallel(benchmark::State& state) {
    int size = state.range(0);
    hatrix::parallel::ParallelMatrixOps ops;
    hatrix::parallel::ParallelHadamardGenerator gen;
    auto matrix = gen.generate_blocked(size);
    
    for (auto _ : state) {
        auto result = ops.multiply_matrices_parallel(matrix, matrix);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * size * sizeof(int) * 3);
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
        std::iota(signals[i].begin(), signals[i].end(), 1.0 + i);
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
        std::iota(signals[i].begin(), signals[i].end(), 1.0 + i);
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
// SCALABILITY BENCHMARKS
//--------------------------------------------------------------------------

static void BM_Scalability_FWHT(benchmark::State& state) {
    int size = state.range(0);
    hatrix::simd::aligned_vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    // Measure different implementations
    std::vector<double> naive_times, avx2_times, parallel_times;
    
    for (auto _ : state) {
        // Naive implementation
        auto start = std::chrono::high_resolution_clock::now();
        auto naive_result = naive_fwht(std::vector<double>(data.begin(), data.end()));
        auto naive_end = std::chrono::high_resolution_clock::now();
        naive_times.push_back(std::chrono::duration<double, std::milli>(naive_end - start).count());
        
        // AVX2 implementation
        start = std::chrono::high_resolution_clock::now();
        auto avx2_result = hatrix::simd::FWHTOptimized::fwht_avx2(data);
        auto avx2_end = std::chrono::high_resolution_clock::now();
        avx2_times.push_back(std::chrono::duration<double, std::milli>(avx2_end - start).count());
        
        // Parallel implementation
        start = std::chrono::high_resolution_clock::now();
        auto parallel_result = hatrix::simd::FWHTOptimized::fwht_parallel(data);
        auto parallel_end = std::chrono::high_resolution_clock::now();
        parallel_times.push_back(std::chrono::duration<double, std::milli>(parallel_end - start).count());
        
        benchmark::DoNotOptimize(naive_result);
        benchmark::DoNotOptimize(avx2_result);
        benchmark::DoNotOptimize(parallel_result);
    }
    
    // Calculate speedup ratios
    double avg_naive = std::accumulate(naive_times.begin(), naive_times.end(), 0.0) / naive_times.size();
    double avg_avx2 = std::accumulate(avx2_times.begin(), avx2_times.end(), 0.0) / avx2_times.size();
    double avg_parallel = std::accumulate(parallel_times.begin(), parallel_times.end(), 0.0) / parallel_times.size();
    
    state.counters["AVX2_Speedup"] = avg_naive / avg_avx2;
    state.counters["Parallel_Speedup"] = avg_naive / avg_parallel;
    state.counters["AVX2_vs_Parallel"] = avg_parallel / avg_avx2;
}

//--------------------------------------------------------------------------
// MEMORY BANDWIDTH BENCHMARKS
//--------------------------------------------------------------------------

static void BM_MemoryBandwidth_MatrixGeneration(benchmark::State& state) {
    int size = state.range(0);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = hadamard::generate_recursive(size);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        state.SetIterationTime(duration.count() / 1e9);
        
        benchmark::DoNotOptimize(result);
    }
    
    size_t bytes_written = state.range(0) * state.range(0) * sizeof(int);
    state.SetBytesProcessed(state.iterations() * bytes_written);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_written, benchmark::Counter::kIsRate);
}

//--------------------------------------------------------------------------
// BENCHMARK REGISTRATION
//--------------------------------------------------------------------------

// FWHT benchmarks
BENCHMARK(BM_FWHT_Naive)->RangeMultiplier(2)->Range(8, 65536)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FWHT_Original)->RangeMultiplier(2)->Range(8, 65536)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FWHT_AVX2)->RangeMultiplier(2)->Range(8, 65536)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FWHT_AVX512)->RangeMultiplier(2)->Range(8, 65536)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FWHT_Parallel)->RangeMultiplier(2)->Range(8, 65536)->Unit(benchmark::kMicrosecond);

// Matrix generation benchmarks
BENCHMARK(BM_Generate_Recursive_Naive)->RangeMultiplier(2)->Range(2, 2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Generate_Iterative_Naive)->RangeMultiplier(2)->Range(2, 2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Generate_Blocked)->RangeMultiplier(2)->Range(2, 2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Generate_Parallel)->RangeMultiplier(2)->Range(2, 2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Generate_SIMD_Parallel)->RangeMultiplier(2)->Range(2, 2048)->Unit(benchmark::kMicrosecond);

// Matrix operation benchmarks
BENCHMARK(BM_Transpose_Naive)->RangeMultiplier(2)->Range(8, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Transpose_SIMD)->RangeMultiplier(2)->Range(8, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatrixVector_Naive)->RangeMultiplier(2)->Range(8, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatrixVector_SIMD)->RangeMultiplier(2)->Range(8, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatrixMatrix_Naive)->RangeMultiplier(2)->Range(8, 256)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatrixMatrix_Parallel)->RangeMultiplier(2)->Range(8, 256)->Unit(benchmark::kMicrosecond);

// Batch processing benchmarks
BENCHMARK(BM_BatchFWHT_Sequential)
    ->Args({10, 64})
    ->Args({10, 256})
    ->Args({10, 1024})
    ->Args({100, 64})
    ->Args({100, 256})
    ->Args({100, 1024})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_BatchFWHT_Parallel)
    ->Args({10, 64})
    ->Args({10, 256})
    ->Args({10, 1024})
    ->Args({100, 64})
    ->Args({100, 256})
    ->Args({100, 1024})
    ->Unit(benchmark::kMicrosecond);

// Scalability benchmarks
BENCHMARK(BM_Scalability_FWHT)->RangeMultiplier(2)->Range(64, 8192)->Unit(benchmark::kMicrosecond);

// Memory bandwidth benchmarks
BENCHMARK(BM_MemoryBandwidth_MatrixGeneration)->RangeMultiplier(2)->Range(64, 2048);

} // namespace perf

//--------------------------------------------------------------------------
// MAIN FUNCTION
//--------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Print system information
    std::cout << "Hatrix Performance Benchmark Suite\n";
    std::cout << "===================================\n";
    std::cout << "SIMD Capabilities:\n";
    std::cout << "  AVX2: " << (hatrix::simd::g_simd_caps.avx2 ? "Yes" : "No") << "\n";
    std::cout << "  AVX-512: " << (hatrix::simd::g_simd_caps.avx512f ? "Yes" : "No") << "\n";
    std::cout << "  FMA: " << (hatrix::simd::g_simd_caps.fma ? "Yes" : "No") << "\n";
    std::cout << "  Max Threads: " << hatrix::simd::g_simd_caps.max_threads << "\n";
    std::cout << "\n";
    
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
