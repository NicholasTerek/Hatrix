// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#include "Hatrix/Matrix.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::size_t> preset_sizes(const std::string& preset) {
    if (preset == "default") {
        return {128, 256, 384, 512, 768, 1024};
    }
    if (preset == "awkward") {
        return {192, 300, 500, 750, 1000};
    }
    if (preset == "large") {
        return {1024, 1536, 2048};
    }
    throw std::invalid_argument("unknown preset: " + preset);
}

int iterations_for_size(std::size_t size) {
    if (size <= 128) {
        return 5;
    }
    if (size <= 256) {
        return 3;
    }
    if (size <= 512) {
        return 2;
    }
    return 1;
}

double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double gflops_for_square_gemm(std::size_t size, double milliseconds) {
    const double flops = 2.0 * static_cast<double>(size) * static_cast<double>(size) *
                         static_cast<double>(size);
    const double seconds = milliseconds / 1000.0;
    return flops / seconds / 1.0e9;
}

std::vector<double> generate_values(std::size_t rows, std::size_t cols, std::uint32_t seed) {
    std::vector<double> values(rows * cols, 0.0);
    std::uint32_t state = seed;

    for (double& value : values) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<double>(state % 1000u) / 1000.0;
    }

    return values;
}

struct BenchmarkResult {
    std::string implementation;
    std::size_t size;
    int iterations;
    double median_ms;
    double gflops;
    double checksum;
};

using MultiplyFunction = Hatrix::Matrix (*)(const Hatrix::Matrix&, const Hatrix::Matrix&);

Hatrix::Matrix multiply_baseline(const Hatrix::Matrix& left, const Hatrix::Matrix& right) {
    return left.multiply(right);
}

Hatrix::Matrix multiply_loop_reordered_impl(
    const Hatrix::Matrix& left,
    const Hatrix::Matrix& right) {
    return left.multiply_loop_reordered(right);
}

BenchmarkResult run_benchmark(
    const std::string& implementation,
    MultiplyFunction multiply_function,
    std::size_t size) {
    const auto left_values = generate_values(size, size, 1u);
    const auto right_values = generate_values(size, size, 2u);

    const Hatrix::Matrix left(size, size, left_values);
    const Hatrix::Matrix right(size, size, right_values);

    std::vector<double> samples;
    samples.reserve(iterations_for_size(size));

    volatile double checksum = 0.0;

    {
        const auto warmup = multiply_function(left, right);
        checksum = warmup.get(0, 0);
    }

    const int iterations = iterations_for_size(size);
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        const auto result = multiply_function(left, right);
        const auto end = Clock::now();

        const auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
        samples.push_back(elapsed.count());
        checksum += result.get(0, 0) + result.get(size - 1, size - 1);
    }

    const double median = median_ms(samples);
    return BenchmarkResult{
        implementation,
        size,
        iterations,
        median,
        gflops_for_square_gemm(size, median),
        checksum,
    };
}

std::vector<std::string> parse_implementations(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--impl") {
        if (std::string(argv[2]) == "all") {
            return {"baseline", "loop-reordered"};
        }
        return {argv[2]};
    }
    if (argc >= 2 && std::string(argv[1]).rfind("--impl=", 0) == 0) {
        const auto value = std::string(argv[1]).substr(7);
        if (value == "all") {
            return {"baseline", "loop-reordered"};
        }
        return {value};
    }
    return {"all"};
}

std::vector<std::size_t> parse_sizes(int argc, char** argv) {
    int start = 1;
    if (argc >= 3 && std::string(argv[1]) == "--impl") {
        start = 3;
    } else if (argc >= 2 && std::string(argv[1]).rfind("--impl=", 0) == 0) {
        start = 2;
    }

    if (argc >= start + 2 && std::string(argv[start]) == "--preset") {
        return preset_sizes(argv[start + 1]);
    }
    if (argc >= start + 1 && std::string(argv[start]).rfind("--preset=", 0) == 0) {
        return preset_sizes(std::string(argv[start]).substr(9));
    }

    if (argc <= start) {
        return preset_sizes("default");
    }

    std::vector<std::size_t> sizes;
    for (int i = start; i < argc; ++i) {
        sizes.push_back(static_cast<std::size_t>(std::stoul(argv[i])));
    }
    return sizes;
}

MultiplyFunction implementation_function(const std::string& implementation) {
    if (implementation == "baseline") {
        return &multiply_baseline;
    }
    if (implementation == "loop-reordered") {
        return &multiply_loop_reordered_impl;
    }
    throw std::invalid_argument("unknown implementation: " + implementation);
}

}  // namespace

int main(int argc, char** argv) {
    const auto implementations = parse_implementations(argc, argv);
    const auto sizes = parse_sizes(argc, argv);

    std::cout << "# Hatrix C++ GEMM Benchmark\n\n";
    std::cout << "| Impl | Size | Iterations | Median ms | GFLOP/s | Checksum |\n";
    std::cout << "| --- | ---: | ---: | ---: | ---: | ---: |\n";

    for (const auto& implementation : implementations) {
        const auto current_implementation =
            implementation == "all"
                ? std::array<std::string, 2>{"baseline", "loop-reordered"}
                : std::array<std::string, 2>{implementation, ""};

        for (const auto& implementation_name : current_implementation) {
            if (implementation_name.empty()) {
                continue;
            }
            for (std::size_t size : sizes) {
                const auto result =
                    run_benchmark(implementation_name, implementation_function(implementation_name), size);
                std::cout << "| " << result.implementation
                  << " | " << result.size
                  << " | " << result.iterations
                  << " | " << std::fixed << std::setprecision(3) << result.median_ms
                  << " | " << std::fixed << std::setprecision(3) << result.gflops
                  << " | " << std::fixed << std::setprecision(3) << result.checksum
                  << " |\n";
            }
        }
    }

    return EXIT_SUCCESS;
}
