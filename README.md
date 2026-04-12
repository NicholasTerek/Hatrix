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
