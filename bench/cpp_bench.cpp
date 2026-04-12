// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#include "Hatrix/Matrix.h"

#include <algorithm>
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
    std::size_t size;
    int iterations;
    double median_ms;
    double gflops;
    double checksum;
};

BenchmarkResult run_benchmark(std::size_t size) {
    const auto left_values = generate_values(size, size, 1u);
    const auto right_values = generate_values(size, size, 2u);

    const Hatrix::Matrix left(size, size, left_values);
    const Hatrix::Matrix right(size, size, right_values);

    std::vector<double> samples;
    samples.reserve(iterations_for_size(size));

    volatile double checksum = 0.0;

    {
        const auto warmup = left.multiply(right);
        checksum = warmup.get(0, 0);
    }

    const int iterations = iterations_for_size(size);
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        const auto result = left.multiply(right);
        const auto end = Clock::now();

        const auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
        samples.push_back(elapsed.count());
        checksum += result.get(0, 0) + result.get(size - 1, size - 1);
    }

    const double median = median_ms(samples);
    return BenchmarkResult{
        size,
        iterations,
        median,
        gflops_for_square_gemm(size, median),
        checksum,
    };
}

std::vector<std::size_t> parse_sizes(int argc, char** argv) {
    if (argc <= 1) {
        return preset_sizes("default");
    }

    std::vector<std::size_t> sizes;
    int start = 1;
    if (argc >= 3 && std::string(argv[1]) == "--preset") {
        return preset_sizes(argv[2]);
    }
    if (argc >= 2 && std::string(argv[1]).rfind("--preset=", 0) == 0) {
        return preset_sizes(std::string(argv[1]).substr(9));
    }

    for (int i = start; i < argc; ++i) {
        sizes.push_back(static_cast<std::size_t>(std::stoul(argv[i])));
    }
    return sizes;
}

}  // namespace

int main(int argc, char** argv) {
    const auto sizes = parse_sizes(argc, argv);

    std::cout << "# Hatrix C++ GEMM Benchmark\n\n";
    std::cout << "| Size | Iterations | Median ms | GFLOP/s | Checksum |\n";
    std::cout << "| --- | ---: | ---: | ---: | ---: |\n";

    for (std::size_t size : sizes) {
        const auto result = run_benchmark(size);
        std::cout << "| " << result.size
                  << " | " << result.iterations
                  << " | " << std::fixed << std::setprecision(3) << result.median_ms
                  << " | " << std::fixed << std::setprecision(3) << result.gflops
                  << " | " << std::fixed << std::setprecision(3) << result.checksum
                  << " |\n";
    }

    return EXIT_SUCCESS;
}
