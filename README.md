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
