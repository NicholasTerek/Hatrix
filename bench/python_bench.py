# Hatrix
# Copyright (c) 2025 Hatrix contributors
# Licensed under the MIT License. See LICENSE for details.

from __future__ import annotations

import argparse
import statistics
import time
from random import Random
from typing import Callable

from hatrix import Matrix

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def preset_sizes(name: str) -> list[int]:
    if name == "default":
        return [128, 256, 384, 512, 768, 1024]
    if name == "awkward":
        return [192, 300, 500, 750, 1000]
    if name == "large":
        return [1024, 1536, 2048]
    raise ValueError(f"unknown preset: {name}")


def iterations_for_size(size: int) -> int:
    if size <= 128:
        return 5
    if size <= 256:
        return 3
    if size <= 512:
        return 2
    return 1


def generate_values(size: int, seed: int) -> list[float]:
    rng = Random(seed)
    return [rng.random() for _ in range(size * size)]


def gflops_for_square_gemm(size: int, milliseconds: float) -> float:
    flops = 2.0 * size * size * size
    return flops / (milliseconds / 1000.0) / 1.0e9


def benchmark_hatrix(size: int) -> tuple[float, float]:
    return benchmark_hatrix_with_impl(size, "baseline")


def benchmark_hatrix_with_impl(size: int, implementation: str) -> tuple[float, float]:
    left = Matrix(size, size, generate_values(size, 1))
    right = Matrix(size, size, generate_values(size, 2))
    multiply: Callable[[Matrix, Matrix], Matrix]
    if implementation == "baseline":
        multiply = lambda a, b: a @ b
    elif implementation == "loop-reordered":
        multiply = lambda a, b: a.multiply_loop_reordered(b)
    elif implementation == "inner-tiled-16":
        multiply = lambda a, b: a.multiply_inner_tiled(b, 16)
    elif implementation == "inner-tiled-32":
        multiply = lambda a, b: a.multiply_inner_tiled(b, 32)
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    samples: list[float] = []
    checksum = 0.0

    warmup = multiply(left, right)
    checksum += warmup[0, 0]

    for _ in range(iterations_for_size(size)):
        start = time.perf_counter_ns()
        result = multiply(left, right)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        samples.append(elapsed_ms)
        checksum += result[0, 0] + result[size - 1, size - 1]

    return statistics.median(samples), checksum


def benchmark_numpy(size: int) -> tuple[float, float] | None:
    if np is None:
        return None

    left = np.array(generate_values(size, 1), dtype=np.float64).reshape(size, size)
    right = np.array(generate_values(size, 2), dtype=np.float64).reshape(size, size)

    samples: list[float] = []
    checksum = 0.0

    warmup = left @ right
    checksum += float(warmup[0, 0])

    for _ in range(iterations_for_size(size)):
        start = time.perf_counter_ns()
        result = left @ right
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        samples.append(elapsed_ms)
        checksum += float(result[0, 0] + result[size - 1, size - 1])

    return statistics.median(samples), checksum


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hatrix from Python.")
    parser.add_argument("sizes", nargs="*", type=int)
    parser.add_argument(
        "--preset",
        choices=["default", "awkward", "large"],
        default="default",
        help="Choose a predefined benchmark size set.",
    )
    parser.add_argument(
        "--impl",
        choices=[
            "baseline",
            "loop-reordered",
            "inner-tiled-16",
            "inner-tiled-32",
            "all",
        ],
        default="all",
        help="Choose which implementation to benchmark.",
    )
    args = parser.parse_args()
    sizes = args.sizes if args.sizes else preset_sizes(args.preset)
    if args.impl == "all":
        implementations = [
            "baseline",
            "loop-reordered",
            "inner-tiled-16",
            "inner-tiled-32",
        ]
    else:
        implementations = [args.impl]

    print("# Hatrix Python GEMM Benchmark\n")
    print("| Impl | Size | Iterations | Hatrix ms | Hatrix GFLOP/s | NumPy ms | NumPy GFLOP/s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for implementation in implementations:
        for size in sizes:
            hatrix_ms, _ = benchmark_hatrix_with_impl(size, implementation)
            hatrix_gflops = gflops_for_square_gemm(size, hatrix_ms)

            numpy_result = benchmark_numpy(size)
            if numpy_result is None:
                numpy_ms = "n/a"
                numpy_gflops = "n/a"
            else:
                numpy_ms_value, _ = numpy_result
                numpy_ms = f"{numpy_ms_value:.3f}"
                numpy_gflops = f"{gflops_for_square_gemm(size, numpy_ms_value):.3f}"

            print(
                f"| {implementation} | {size} | {iterations_for_size(size)} | {hatrix_ms:.3f} | "
                f"{hatrix_gflops:.3f} | {numpy_ms} | {numpy_gflops} |"
            )

    if np is None:
        print("\nNumPy is not installed, so the NumPy comparison columns are unavailable.")


if __name__ == "__main__":
    main()
