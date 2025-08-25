# Hatrix

A header-only C++17 library for **Hadamard matrices** and the **Fast Walsh–Hadamard Transform (FWHT)**.  
Focus: clean math, fast transforms, reliable validation.

---

## 1. Core Hadamard
- [ ] Sylvester recursive construction
- [ ] Iterative expansion
- [ ] Walsh orderings (natural, sequency, dyadic)
- [ ] Validation: orthogonality + Hadamard property
- [ ] Matrix operations: transpose, multiply

---

## 2. Transforms
- [ ] Fast Walsh–Hadamard Transform (FWHT)
- [ ] Inverse FWHT (scaling)
- [ ] Normalization modes
- [ ] Statistical analysis (determinant, sequency, condition number)

---

## 3. Utilities
- [ ] Pretty-printing (compact, verbose, LaTeX, CSV, binary)
- [ ] Serialization + deserialization
- [ ] File I/O (save/load)
- [ ] Random Hadamard-like matrices
- [ ] Benchmark helper

---

## 4. Performance
- [ ] Microbenchmarks (2^k sizes)
- [ ] SIMD kernels (AVX2/AVX-512)
- [ ] Multithreaded FWHT
- [ ] Cache-aware tiling
- [ ] Compare against FFTW WHT / naïve impl

---

## 5. Surroundings (nice-to-have)
- [ ] Minimal README usage examples
- [ ] Unit tests (generation, FWHT roundtrip, orthogonality)
- [ ] GitHub Actions CI (build + test)
- [ ] Python bindings (pybind11) in v1.1
