# Hatrix Matrix Multiplication Phases

This project is Python-first, with the performance-critical implementation written in C++.

## Phase 1

- Add a C++ GEMM benchmark harness for raw kernel timing.
- Add a Python GEMM benchmark harness for public-API timing.
- Compare Python performance to NumPy when NumPy is available.
- Keep correctness checks separate from benchmarks.

## Planned follow-up phases

1. Baseline naive GEMM documentation and benchmark capture
2. Loop-order experiments
3. Inner-dimension tiling
4. Multi-dimension tiling
5. Multithreading
6. SIMD and register blocking
7. Benchmark summary and regression tracking

## Suggested issue and PR sequence

1. Benchmark harness and baseline GEMM
2. Loop-order optimization
3. Inner-dimension tiling
4. Multi-dimension tiling
5. Parallel GEMM
6. SIMD and register blocking
7. Benchmark result documentation
