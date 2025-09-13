//======================================================================
// benchmark_hadamard_gbench.cpp
//----------------------------------------------------------------------
// Google Benchmark performance tests for the Hadamard matrix library.
// Comprehensive benchmarking across various sizes and operations.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================

#include <benchmark/benchmark.h>
#include "../Hatrix/hadamard_matrix.hpp"
#include <vector>
#include <random>

//--------------------------------------------------------------------------
// HELPER FUNCTIONS
//--------------------------------------------------------------------------
template<typename Func>
void BM_Generation(benchmark::State& state, Func&& gen_func) {
    int size = state.range(0);
    
    for (auto _ : state) {
        auto H = gen_func(size);
        benchmark::DoNotOptimize(H);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

template<typename Func>
void BM_Transform(benchmark::State& state, Func&& transform_func) {
    int size = state.range(0);
    hadamard::dvector_t data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    for (auto _ : state) {
        auto result = transform_func(data);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * sizeof(double) * 2); // Input + output
}

template<typename Func>
void BM_Operation(benchmark::State& state, Func&& op_func) {
    int size = state.range(0);
    auto H = hadamard::generate_recursive(size);
    
    for (auto _ : state) {
        auto result = op_func(H);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int) * 2); // Input + output
}

//--------------------------------------------------------------------------
// GENERATION BENCHMARKS
//--------------------------------------------------------------------------
static void BM_GenerateRecursive(benchmark::State& state) {
    BM_Generation(state, [](int n) { return hadamard::generate_recursive(n); });
}

static void BM_GenerateIterative(benchmark::State& state) {
    BM_Generation(state, [](int n) { return hadamard::generate_iterative(n); });
}

static void BM_GenerateWalshNatural(benchmark::State& state) {
    BM_Generation(state, [](int n) { 
        return hadamard::generate_walsh(n, hadamard::ordering_t::NATURAL); 
    });
}

static void BM_GenerateWalshSequency(benchmark::State& state) {
    BM_Generation(state, [](int n) { 
        return hadamard::generate_walsh(n, hadamard::ordering_t::SEQUENCY); 
    });
}

static void BM_GenerateWalshDyadic(benchmark::State& state) {
    BM_Generation(state, [](int n) { 
        return hadamard::generate_walsh(n, hadamard::ordering_t::DYADIC); 
    });
}

//--------------------------------------------------------------------------
// TRANSFORM BENCHMARKS
//--------------------------------------------------------------------------
static void BM_FWHT(benchmark::State& state) {
    BM_Transform(state, [](const hadamard::dvector_t& data) {
        return hadamard::fwht(data);
    });
}

static void BM_IFWHT(benchmark::State& state) {
    BM_Transform(state, [](const hadamard::dvector_t& data) {
        return hadamard::ifwht(data);
    });
}

static void BM_FWHT_RoundTrip(benchmark::State& state) {
    int size = state.range(0);
    hadamard::dvector_t data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    for (auto _ : state) {
        auto transformed = hadamard::fwht(data);
        auto reconstructed = hadamard::ifwht(transformed);
        benchmark::DoNotOptimize(reconstructed);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * sizeof(double) * 4); // Input + transform + inverse + output
}

//--------------------------------------------------------------------------
// MATRIX OPERATION BENCHMARKS
//--------------------------------------------------------------------------
static void BM_Transpose(benchmark::State& state) {
    BM_Operation(state, [](const hadamard::matrix_t& H) {
        return hadamard::transpose(H);
    });
}

static void BM_MatrixVectorMultiply(benchmark::State& state) {
    int size = state.range(0);
    auto H = hadamard::generate_recursive(size);
    hadamard::vector_t v(size);
    std::iota(v.begin(), v.end(), 1);
    
    for (auto _ : state) {
        auto result = hadamard::multiply(H, v);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int) + size * sizeof(int));
}

static void BM_MatrixMatrixMultiply(benchmark::State& state) {
    BM_Operation(state, [](const hadamard::matrix_t& H) {
        return hadamard::multiply(H, H);
    });
}

static void BM_IsOrthogonal(benchmark::State& state) {
    BM_Operation(state, [](const hadamard::matrix_t& H) {
        return hadamard::is_orthogonal(H);
    });
}

static void BM_IsHadamard(benchmark::State& state) {
    BM_Operation(state, [](const hadamard::matrix_t& H) {
        return hadamard::is_hadamard(H);
    });
}

//--------------------------------------------------------------------------
// SERIALIZATION BENCHMARKS
//--------------------------------------------------------------------------
static void BM_Serialize(benchmark::State& state) {
    int size = state.range(0);
    auto H = hadamard::generate_recursive(size);
    
    for (auto _ : state) {
        std::string data = hadamard::serialize(H, hadamard::format_t::COMPACT);
        benchmark::DoNotOptimize(data);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

static void BM_Deserialize(benchmark::State& state) {
    int size = state.range(0);
    auto H = hadamard::generate_recursive(size);
    std::string serialized = hadamard::serialize(H, hadamard::format_t::COMPACT);
    
    for (auto _ : state) {
        auto deserialized = hadamard::deserialize(serialized, hadamard::format_t::COMPACT);
        benchmark::DoNotOptimize(deserialized);
    }
    
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * size * sizeof(int));
}

//--------------------------------------------------------------------------
// PROPERTIES ANALYSIS BENCHMARKS
//--------------------------------------------------------------------------
static void BM_AnalyzeProperties(benchmark::State& state) {
    BM_Operation(state, [](const hadamard::matrix_t& H) {
        return hadamard::analyze_properties(H);
    });
}

//--------------------------------------------------------------------------
// BATCH PROCESSING BENCHMARKS
//--------------------------------------------------------------------------
static void BM_BatchFWHT(benchmark::State& state) {
    int size = state.range(0);
    int batch_size = state.range(1);
    
    std::vector<hadamard::dvector_t> signals(batch_size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < batch_size; ++i) {
        signals[i].resize(size);
        for (int j = 0; j < size; ++j) {
            signals[i][j] = dist(rng);
        }
    }
    
    for (auto _ : state) {
        for (int i = 0; i < batch_size; ++i) {
            auto transformed = hadamard::fwht(signals[i]);
            benchmark::DoNotOptimize(transformed);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(double) * 2);
}

//--------------------------------------------------------------------------
// BENCHMARK REGISTRATION
//--------------------------------------------------------------------------
// Generation benchmarks
BENCHMARK(BM_GenerateRecursive)->RangeMultiplier(2)->Range(2, 512)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GenerateIterative)->RangeMultiplier(2)->Range(2, 2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GenerateWalshNatural)->RangeMultiplier(2)->Range(2, 512)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GenerateWalshSequency)->RangeMultiplier(2)->Range(2, 512)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GenerateWalshDyadic)->RangeMultiplier(2)->Range(2, 512)->Unit(benchmark::kMicrosecond);

// Transform benchmarks
BENCHMARK(BM_FWHT)->RangeMultiplier(2)->Range(2, 4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_IFWHT)->RangeMultiplier(2)->Range(2, 4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FWHT_RoundTrip)->RangeMultiplier(2)->Range(2, 4096)->Unit(benchmark::kMicrosecond);

// Matrix operation benchmarks
BENCHMARK(BM_Transpose)->RangeMultiplier(2)->Range(2, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatrixVectorMultiply)->RangeMultiplier(2)->Range(2, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MatrixMatrixMultiply)->RangeMultiplier(2)->Range(2, 256)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_IsOrthogonal)->RangeMultiplier(2)->Range(2, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_IsHadamard)->RangeMultiplier(2)->Range(2, 1024)->Unit(benchmark::kMicrosecond);

// Serialization benchmarks
BENCHMARK(BM_Serialize)->RangeMultiplier(2)->Range(2, 512)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Deserialize)->RangeMultiplier(2)->Range(2, 512)->Unit(benchmark::kMicrosecond);

// Properties analysis benchmarks
BENCHMARK(BM_AnalyzeProperties)->RangeMultiplier(2)->Range(2, 512)->Unit(benchmark::kMicrosecond);

// Batch processing benchmarks
BENCHMARK(BM_BatchFWHT)
    ->Args({64, 10})
    ->Args({64, 100})
    ->Args({256, 10})
    ->Args({256, 100})
    ->Args({1024, 10})
    ->Unit(benchmark::kMicrosecond);

//--------------------------------------------------------------------------
// CUSTOM BENCHMARK ARGUMENTS
//--------------------------------------------------------------------------
// Compare recursive vs iterative generation
BENCHMARK(BM_GenerateRecursive)->Name("GenerateRecursive")->RangeMultiplier(2)->Range(2, 512);
BENCHMARK(BM_GenerateIterative)->Name("GenerateIterative")->RangeMultiplier(2)->Range(2, 512);

// Compare different Walsh orderings
BENCHMARK(BM_GenerateWalshNatural)->Name("WalshNatural")->RangeMultiplier(2)->Range(2, 512);
BENCHMARK(BM_GenerateWalshSequency)->Name("WalshSequency")->RangeMultiplier(2)->Range(2, 512);
BENCHMARK(BM_GenerateWalshDyadic)->Name("WalshDyadic")->RangeMultiplier(2)->Range(2, 512);

//--------------------------------------------------------------------------
// MAIN FUNCTION
//--------------------------------------------------------------------------
int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
