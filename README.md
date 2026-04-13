# Hatrix

Minimal C++ matrix library with a Python binding.

**What is here**

- A compiled C++ shared library with a small `Matrix` type.
- A C ABI layer so Python can call into the library without extra binding dependencies.
- A lightweight Python package in `python/hatrix`.
- Basic Hadamard utilities: `transpose`, `kronecker`, `normalize`, `is_hadamard`, and `sylvester`.

**Build**

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

**Benchmarks**

Build the native benchmark:

```bash
cmake --build build --target hatrix_bench
./build/hatrix_bench
```

Run the Python end-to-end benchmark:

```bash
PYTHONPATH=python python3 bench/python_bench.py
```

Benchmark presets:

- `default`: `128 256 384 512 768 1024`
- `awkward`: `192 300 500 750 1000`
- `large`: `1024 1536 2048`

Examples:

```bash
./build/hatrix_bench --preset default
./build/hatrix_bench --preset awkward
./build/hatrix_bench 256 512 1024
```

```bash
PYTHONPATH=python python3 bench/python_bench.py --preset default
PYTHONPATH=python python3 bench/python_bench.py --preset awkward
PYTHONPATH=python python3 bench/python_bench.py 256 512 1024
```

If NumPy is installed, the Python benchmark will include a NumPy comparison column.

Tracked presets:

- `default`: the main baseline to track in the README over time
- `awkward`: non-power-of-two sizes to catch edge and tiling regressions
- `large`: opt-in stress sizes, useful but expensive with the current implementation

Current baseline on this machine for `default`

C++ benchmark:

```text
| Size | Iterations | Median ms | GFLOP/s |
| 128  | 5          | 8.729     | 0.480   |
| 256  | 3          | 84.978    | 0.395   |
| 384  | 2          | 318.562   | 0.355   |
| 512  | 2          | 701.271   | 0.383   |
| 768  | 1          | 2098.212  | 0.432   |
| 1024 | 1          | 5665.702  | 0.379   |
```

Python benchmark:

```text
| Size | Iterations | Hatrix ms | Hatrix GFLOP/s | NumPy ms | NumPy GFLOP/s |
| 128  | 5          | 8.844     | 0.474          | 0.077    | 54.309        |
| 256  | 3          | 72.643    | 0.462          | 4.996    | 6.716         |
| 384  | 2          | 261.339   | 0.433          | 2.399    | 47.211        |
| 512  | 2          | 681.467   | 0.394          | 2.868    | 93.584        |
| 768  | 1          | 2406.081  | 0.377          | 9.499    | 95.374        |
| 1024 | 1          | 5863.597  | 0.366          | 9.536    | 225.186       |
```

These numbers are machine-specific and should be treated as a local baseline, not a universal claim.

Current baseline on this machine for `awkward`

C++ benchmark:

```text
| Size | Iterations | Median ms | GFLOP/s |
| 192  | 3          | 28.959    | 0.489   |
| 300  | 2          | 138.611   | 0.390   |
| 500  | 2          | 638.482   | 0.392   |
| 750  | 1          | 2257.694  | 0.374   |
| 1000 | 1          | 5119.626  | 0.391   |
```

Python benchmark:

```text
| Size | Iterations | Hatrix ms | Hatrix GFLOP/s | NumPy ms | NumPy GFLOP/s |
| 192  | 3          | 29.683    | 0.477          | 2.887    | 4.903         |
| 300  | 2          | 116.544   | 0.463          | 1.856    | 29.101        |
| 500  | 2          | 657.801   | 0.380          | 4.163    | 60.048        |
| 750  | 1          | 2031.197  | 0.415          | 17.982   | 46.923        |
| 1000 | 1          | 5025.094  | 0.398          | 9.323    | 214.515       |
```

Current issue `#2` snapshot: `baseline` vs `loop-reordered` on `default`

C++ benchmark:

```text
| Impl           | Size | Median ms | GFLOP/s |
| baseline       | 128  | 8.869     | 0.473   |
| baseline       | 256  | 90.040    | 0.373   |
| baseline       | 384  | 285.146   | 0.397   |
| baseline       | 512  | 754.249   | 0.356   |
| baseline       | 768  | 2260.257  | 0.401   |
| baseline       | 1024 | 5477.922  | 0.392   |
| loop-reordered | 128  | 8.678     | 0.483   |
| loop-reordered | 256  | 69.575    | 0.482   |
| loop-reordered | 384  | 234.745   | 0.482   |
| loop-reordered | 512  | 563.243   | 0.477   |
| loop-reordered | 768  | 2139.046  | 0.424   |
| loop-reordered | 1024 | 4649.936  | 0.462   |
```

Python benchmark:

```text
| Impl           | Size | Hatrix ms | Hatrix GFLOP/s | NumPy ms | NumPy GFLOP/s |
| baseline       | 128  | 8.898     | 0.471          | 0.080    | 52.369        |
| baseline       | 256  | 72.926    | 0.460          | 1.830    | 18.332        |
| baseline       | 384  | 251.138   | 0.451          | 12.993   | 8.716         |
| baseline       | 512  | 639.153   | 0.420          | 11.992   | 22.384        |
| baseline       | 768  | 2395.825  | 0.378          | 11.983   | 75.601        |
| baseline       | 1024 | 5385.624  | 0.399          | 15.063   | 142.564       |
| loop-reordered | 128  | 14.396    | 0.291          | 0.060    | 70.445        |
| loop-reordered | 256  | 76.948    | 0.436          | 2.954    | 11.359        |
| loop-reordered | 384  | 247.680   | 0.457          | 9.493    | 11.930        |
| loop-reordered | 512  | 638.300   | 0.421          | 3.382    | 79.361        |
| loop-reordered | 768  | 2034.198  | 0.445          | 13.991   | 64.752        |
| loop-reordered | 1024 | 4983.427  | 0.431          | 10.842   | 198.063       |
```

Current issue `#3` snapshot: inner-dimension tiling on `default`

C++ benchmark:

```text
| Impl           | Size | Median ms | GFLOP/s |
| baseline       | 256  | 78.250    | 0.429   |
| loop-reordered | 256  | 77.949    | 0.430   |
| inner-tiled-16 | 256  | 71.024    | 0.472   |
| inner-tiled-32 | 256  | 68.221    | 0.492   |
| baseline       | 512  | 700.405   | 0.383   |
| loop-reordered | 512  | 692.512   | 0.388   |
| inner-tiled-16 | 512  | 607.456   | 0.442   |
| inner-tiled-32 | 512  | 622.342   | 0.431   |
| baseline       | 1024 | 5584.051  | 0.385   |
| loop-reordered | 1024 | 5160.864  | 0.416   |
| inner-tiled-16 | 1024 | 4950.435  | 0.434   |
| inner-tiled-32 | 1024 | 4893.797  | 0.439   |
```

Python benchmark:

```text
| Impl           | Size | Hatrix ms | Hatrix GFLOP/s |
| baseline       | 256  | 72.881    | 0.460          |
| loop-reordered | 256  | 70.578    | 0.475          |
| inner-tiled-16 | 256  | 78.474    | 0.428          |
| inner-tiled-32 | 256  | 78.349    | 0.428          |
| baseline       | 512  | 658.322   | 0.408          |
| loop-reordered | 512  | 587.295   | 0.457          |
| inner-tiled-16 | 512  | 617.828   | 0.434          |
| inner-tiled-32 | 512  | 615.303   | 0.436          |
| baseline       | 1024 | 5620.385  | 0.382          |
| loop-reordered | 1024 | 4729.890  | 0.454          |
| inner-tiled-16 | 1024 | 4916.125  | 0.437          |
| inner-tiled-32 | 1024 | 4842.155  | 0.443          |
```

Summary:

- inner tiling clearly improves over the original baseline
- `inner-tiled-16` and `inner-tiled-32` are the only tiled variants worth keeping
- on the native benchmark, tiling is the current best approach at the larger tracked sizes
- on the Python end-to-end benchmark, loop reordering is still slightly ahead at `512` and `1024`

**Use from C++**

Headers live in `include/Hatrix`, and the shared library target is `hatrix`.

**Use from Python**

Build first, then point `PYTHONPATH` at the Python binding folder:

```bash
PYTHONPATH=python python3
```

Example:

```python
from hatrix import Matrix

a = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
b = Matrix(2, 2, [5.0, 6.0, 7.0, 8.0])

print(a.add(b).to_list())
print(a.multiply(b).to_list())
print(Matrix.sylvester(2).to_list())
```

Run the example:

```bash
PYTHONPATH=python python3 examples/python_example.py
```
