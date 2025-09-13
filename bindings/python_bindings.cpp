//======================================================================
// python_bindings.cpp
//----------------------------------------------------------------------
// Python bindings for the Hadamard matrix library using pybind11.
// Provides a complete Python interface to all library functionality.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include "../Hatrix/hadamard_matrix.hpp"
#include "../Hatrix/hatrix_simd.hpp"
#include "../Hatrix/hatrix_parallel.hpp"
#include <vector>
#include <string>

namespace py = pybind11;

//--------------------------------------------------------------------------
// HELPER FUNCTIONS FOR NUMPY COMPATIBILITY
//--------------------------------------------------------------------------
py::array_t<int> matrix_to_numpy(const hadamard::matrix_t& matrix) {
    if (matrix.empty()) {
        return py::array_t<int>({0, 0});
    }
    
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    auto result = py::array_t<int>({rows, cols});
    auto buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ptr[i * cols + j] = matrix[i][j];
        }
    }
    
    return result;
}

hadamard::matrix_t numpy_to_matrix(py::array_t<int> array) {
    auto buf = array.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Array must be 2-dimensional");
    }
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    int* ptr = static_cast<int*>(buf.ptr);
    
    hadamard::matrix_t matrix(rows, std::vector<int>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = ptr[i * cols + j];
        }
    }
    
    return matrix;
}

py::array_t<double> vector_to_numpy(const hadamard::dvector_t& vector) {
    auto result = py::array_t<double>(vector.size());
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < vector.size(); ++i) {
        ptr[i] = vector[i];
    }
    
    return result;
}

hadamard::dvector_t numpy_to_vector(py::array_t<double> array) {
    auto buf = array.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Array must be 1-dimensional");
    }
    
    size_t size = buf.shape[0];
    double* ptr = static_cast<double*>(buf.ptr);
    
    hadamard::dvector_t vector(size);
    for (size_t i = 0; i < size; ++i) {
        vector[i] = ptr[i];
    }
    
    return vector;
}

//--------------------------------------------------------------------------
// PYTHON BINDINGS
//--------------------------------------------------------------------------
PYBIND11_MODULE(hatrix, m) {
    m.doc() = "Hadamard matrix library with Fast Walsh-Hadamard Transform";
    
    //----------------------------------------------------------------------
    // ENUMS
    //----------------------------------------------------------------------
    py::enum_<hadamard::ordering_t>(m, "Ordering")
        .value("NATURAL", hadamard::ordering_t::NATURAL)
        .value("SEQUENCY", hadamard::ordering_t::SEQUENCY)
        .value("DYADIC", hadamard::ordering_t::DYADIC);
    
    py::enum_<hadamard::format_t>(m, "Format")
        .value("COMPACT", hadamard::format_t::COMPACT)
        .value("VERBOSE", hadamard::format_t::VERBOSE)
        .value("LATEX", hadamard::format_t::LATEX)
        .value("CSV", hadamard::format_t::CSV)
        .value("BINARY", hadamard::format_t::BINARY);
    
    //----------------------------------------------------------------------
    // MATRIX GENERATION FUNCTIONS
    //----------------------------------------------------------------------
    m.def("generate_recursive", [](int n) -> py::array_t<int> {
        auto matrix = hadamard::generate_recursive(n);
        return matrix_to_numpy(matrix);
    }, py::arg("n"), "Generate Hadamard matrix using recursive Sylvester construction");
    
    m.def("generate_iterative", [](int n) -> py::array_t<int> {
        auto matrix = hadamard::generate_iterative(n);
        return matrix_to_numpy(matrix);
    }, py::arg("n"), "Generate Hadamard matrix using iterative expansion");
    
    m.def("generate_walsh", [](int n, hadamard::ordering_t order = hadamard::ordering_t::NATURAL) -> py::array_t<int> {
        auto matrix = hadamard::generate_walsh(n, order);
        return matrix_to_numpy(matrix);
    }, py::arg("n"), py::arg("order") = hadamard::ordering_t::NATURAL, 
       "Generate Walsh-ordered Hadamard matrix");
    
    m.def("generate_random_binary", [](int n, int seed = 42) -> py::array_t<int> {
        auto matrix = hadamard::generate_random_binary(n, seed);
        return matrix_to_numpy(matrix);
    }, py::arg("n"), py::arg("seed") = 42, 
       "Generate random binary matrix for testing");
    
    //----------------------------------------------------------------------
    // TRANSFORM FUNCTIONS
    //----------------------------------------------------------------------
    m.def("fwht", [](py::array_t<double> data) -> py::array_t<double> {
        auto vector = numpy_to_vector(data);
        auto result = hadamard::fwht(vector);
        return vector_to_numpy(result);
    }, py::arg("data"), "Fast Walsh-Hadamard Transform (forward)");
    
    m.def("fwht_optimized", [](py::array_t<double> data) -> py::array_t<double> {
        auto vector = numpy_to_vector(data);
        hatrix::simd::aligned_vector<double> aligned_data(vector.begin(), vector.end());
        auto result = hatrix::simd::FWHTOptimized::fwht_parallel(aligned_data);
        return vector_to_numpy(hatrix::dvector_t(result.begin(), result.end()));
    }, py::arg("data"), "Optimized Fast Walsh-Hadamard Transform with SIMD and threading");
    
    m.def("ifwht", [](py::array_t<double> data) -> py::array_t<double> {
        auto vector = numpy_to_vector(data);
        auto result = hadamard::ifwht(vector);
        return vector_to_numpy(result);
    }, py::arg("data"), "Inverse Fast Walsh-Hadamard Transform");
    
    m.def("ifwht_optimized", [](py::array_t<double> data) -> py::array_t<double> {
        auto vector = numpy_to_vector(data);
        hatrix::simd::aligned_vector<double> aligned_data(vector.begin(), vector.end());
        auto result = hatrix::simd::FWHTOptimized::fwht_parallel(aligned_data);
        return vector_to_numpy(hatrix::dvector_t(result.begin(), result.end()));
    }, py::arg("data"), "Optimized Inverse Fast Walsh-Hadamard Transform");
    
    //----------------------------------------------------------------------
    // MATRIX OPERATIONS
    //----------------------------------------------------------------------
    m.def("transpose", [](py::array_t<int> matrix) -> py::array_t<int> {
        auto mat = numpy_to_matrix(matrix);
        auto result = hadamard::transpose(mat);
        return matrix_to_numpy(result);
    }, py::arg("matrix"), "Matrix transpose");
    
    m.def("transpose_optimized", [](py::array_t<int> matrix) -> py::array_t<int> {
        auto buf = matrix.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        int* data = static_cast<int*>(buf.ptr);
        
        hatrix::parallel::CacheOptimizedMatrix<int> cache_matrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cache_matrix(i, j) = data[i * cols + j];
            }
        }
        
        hatrix::parallel::ParallelMatrixOps ops;
        auto result = ops.transpose_parallel(cache_matrix);
        
        auto output = py::array_t<int>({cols, rows});
        auto out_buf = output.request();
        int* out_ptr = static_cast<int*>(out_buf.ptr);
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                out_ptr[i * rows + j] = result(i, j);
            }
        }
        return output;
    }, py::arg("matrix"), "Optimized matrix transpose with SIMD and threading");
    
    m.def("multiply", [](py::array_t<int> matrix, py::array_t<int> vector) -> py::array_t<int> {
        auto mat = numpy_to_matrix(matrix);
        hadamard::vector_t vec(vector.size());
        auto buf = vector.request();
        int* ptr = static_cast<int*>(buf.ptr);
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = ptr[i];
        }
        auto result = hadamard::multiply(mat, vec);
        
        auto output = py::array_t<int>(result.size());
        auto out_buf = output.request();
        int* out_ptr = static_cast<int*>(out_buf.ptr);
        for (size_t i = 0; i < result.size(); ++i) {
            out_ptr[i] = result[i];
        }
        return output;
    }, py::arg("matrix"), py::arg("vector"), "Matrix-vector multiplication");
    
    m.def("multiply_optimized", [](py::array_t<int> matrix, py::array_t<int> vector) -> py::array_t<int> {
        auto buf = matrix.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        int* data = static_cast<int*>(buf.ptr);
        
        hatrix::parallel::CacheOptimizedMatrix<int> cache_matrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cache_matrix(i, j) = data[i * cols + j];
            }
        }
        
        std::vector<int> vec(vector.size());
        auto vec_buf = vector.request();
        int* vec_ptr = static_cast<int*>(vec_buf.ptr);
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = vec_ptr[i];
        }
        
        hatrix::parallel::ParallelMatrixOps ops;
        auto result = ops.multiply_vector_parallel(cache_matrix, vec);
        
        auto output = py::array_t<int>(result.size());
        auto out_buf = output.request();
        int* out_ptr = static_cast<int*>(out_buf.ptr);
        for (size_t i = 0; i < result.size(); ++i) {
            out_ptr[i] = result[i];
        }
        return output;
    }, py::arg("matrix"), py::arg("vector"), "Optimized matrix-vector multiplication with SIMD and threading");
    
    m.def("multiply_matrices", [](py::array_t<int> A, py::array_t<int> B) -> py::array_t<int> {
        auto mat_A = numpy_to_matrix(A);
        auto mat_B = numpy_to_matrix(B);
        auto result = hadamard::multiply(mat_A, mat_B);
        return matrix_to_numpy(result);
    }, py::arg("A"), py::arg("B"), "Matrix-matrix multiplication");
    
    m.def("multiply_matrices_optimized", [](py::array_t<int> A, py::array_t<int> B) -> py::array_t<int> {
        auto buf_A = A.request();
        auto buf_B = B.request();
        int rows_A = buf_A.shape[0];
        int cols_A = buf_A.shape[1];
        int cols_B = buf_B.shape[1];
        int* data_A = static_cast<int*>(buf_A.ptr);
        int* data_B = static_cast<int*>(buf_B.ptr);
        
        hatrix::parallel::CacheOptimizedMatrix<int> cache_A(rows_A, cols_A);
        hatrix::parallel::CacheOptimizedMatrix<int> cache_B(cols_A, cols_B);
        
        for (int i = 0; i < rows_A; ++i) {
            for (int j = 0; j < cols_A; ++j) {
                cache_A(i, j) = data_A[i * cols_A + j];
            }
        }
        
        for (int i = 0; i < cols_A; ++i) {
            for (int j = 0; j < cols_B; ++j) {
                cache_B(i, j) = data_B[i * cols_B + j];
            }
        }
        
        hatrix::parallel::ParallelMatrixOps ops;
        auto result = ops.multiply_matrices_parallel(cache_A, cache_B);
        
        auto output = py::array_t<int>({rows_A, cols_B});
        auto out_buf = output.request();
        int* out_ptr = static_cast<int*>(out_buf.ptr);
        for (int i = 0; i < rows_A; ++i) {
            for (int j = 0; j < cols_B; ++j) {
                out_ptr[i * cols_B + j] = result(i, j);
            }
        }
        return output;
    }, py::arg("A"), py::arg("B"), "Optimized matrix-matrix multiplication with SIMD and threading");
    
    //----------------------------------------------------------------------
    // VALIDATION FUNCTIONS
    //----------------------------------------------------------------------
    m.def("is_hadamard", [](py::array_t<int> matrix) -> bool {
        auto mat = numpy_to_matrix(matrix);
        return hadamard::is_hadamard(mat);
    }, py::arg("matrix"), "Check if matrix is a valid Hadamard matrix");
    
    m.def("is_orthogonal", [](py::array_t<int> matrix) -> bool {
        auto mat = numpy_to_matrix(matrix);
        return hadamard::is_orthogonal(mat);
    }, py::arg("matrix"), "Check if matrix is orthogonal");
    
    m.def("validate_matrix", [](py::array_t<int> matrix) -> std::vector<std::string> {
        auto mat = numpy_to_matrix(matrix);
        return hadamard::validate_matrix(mat);
    }, py::arg("matrix"), "Comprehensive matrix validation");
    
    m.def("validate_order", [](int n) -> void {
        hadamard::validate_order(n);
    }, py::arg("n"), "Validate that order is a positive power of 2");
    
    //----------------------------------------------------------------------
    // PROPERTIES ANALYSIS
    //----------------------------------------------------------------------
    m.def("analyze_properties", [](py::array_t<int> matrix) -> py::dict {
        auto mat = numpy_to_matrix(matrix);
        auto props = hadamard::analyze_properties(mat);
        
        py::dict result;
        for (const auto& [key, value] : props) {
            result[key.c_str()] = value;
        }
        return result;
    }, py::arg("matrix"), "Analyze matrix properties");
    
    //----------------------------------------------------------------------
    // SERIALIZATION FUNCTIONS
    //----------------------------------------------------------------------
    m.def("serialize", [](py::array_t<int> matrix, hadamard::format_t fmt = hadamard::format_t::COMPACT) -> std::string {
        auto mat = numpy_to_matrix(matrix);
        return hadamard::serialize(mat, fmt);
    }, py::arg("matrix"), py::arg("format") = hadamard::format_t::COMPACT, 
       "Serialize matrix to string");
    
    m.def("deserialize", [](const std::string& data, hadamard::format_t fmt = hadamard::format_t::COMPACT) -> py::array_t<int> {
        auto matrix = hadamard::deserialize(data, fmt);
        return matrix_to_numpy(matrix);
    }, py::arg("data"), py::arg("format") = hadamard::format_t::COMPACT, 
       "Deserialize string to matrix");
    
    //----------------------------------------------------------------------
    // FILE I/O FUNCTIONS
    //----------------------------------------------------------------------
    m.def("save_to_file", [](py::array_t<int> matrix, const std::string& filename, 
                           hadamard::format_t fmt = hadamard::format_t::COMPACT) -> void {
        auto mat = numpy_to_matrix(matrix);
        hadamard::save_to_file(mat, filename, fmt);
    }, py::arg("matrix"), py::arg("filename"), py::arg("format") = hadamard::format_t::COMPACT, 
       "Save matrix to file");
    
    m.def("load_from_file", [](const std::string& filename, hadamard::format_t fmt = hadamard::format_t::COMPACT) -> py::array_t<int> {
        auto matrix = hadamard::load_from_file(filename, fmt);
        return matrix_to_numpy(matrix);
    }, py::arg("filename"), py::arg("format") = hadamard::format_t::COMPACT, 
       "Load matrix from file");
    
    //----------------------------------------------------------------------
    // UTILITY FUNCTIONS
    //----------------------------------------------------------------------
    m.def("print_matrix", [](py::array_t<int> matrix, hadamard::format_t fmt = hadamard::format_t::VERBOSE) -> std::string {
        auto mat = numpy_to_matrix(matrix);
        std::ostringstream oss;
        hadamard::print(mat, oss, fmt);
        return oss.str();
    }, py::arg("matrix"), py::arg("format") = hadamard::format_t::VERBOSE, 
       "Pretty-print matrix to string");
    
    m.def("binary_to_gray", &hadamard::binary_to_gray, py::arg("n"), "Convert binary to Gray code");
    m.def("gray_to_binary", &hadamard::gray_to_binary, py::arg("n"), "Convert Gray code to binary");
    
    //----------------------------------------------------------------------
    // BENCHMARKING
    //----------------------------------------------------------------------
    m.def("benchmark", [](py::function func, int iterations = 1) -> double {
        return hadamard::benchmark([&func]() { func(); }, iterations);
    }, py::arg("func"), py::arg("iterations") = 1, 
       "Benchmark a Python function");
    
    //----------------------------------------------------------------------
    // MODULE METADATA
    //----------------------------------------------------------------------
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Nicholas Terek";
    
    // Add version info as module-level constants
    m.attr("LIBRARY_VERSION") = hadamard::library_version;
    m.attr("LIBRARY_AUTHOR") = hadamard::library_author;
    
    //----------------------------------------------------------------------
    // HIGH-PERFORMANCE FUNCTIONS
    //----------------------------------------------------------------------
    m.def("create_hadamard_optimized", [](int n, const std::string& method = "parallel") -> py::array_t<int> {
        hatrix::parallel::ParallelHadamardGenerator generator;
        
        if (method == "blocked") {
            auto result = generator.generate_blocked(n);
            auto output = py::array_t<int>({n, n});
            auto buf = output.request();
            int* ptr = static_cast<int*>(buf.ptr);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    ptr[i * n + j] = result(i, j);
                }
            }
            return output;
        } else if (method == "parallel") {
            auto result = generator.generate_parallel(n);
            auto output = py::array_t<int>({n, n});
            auto buf = output.request();
            int* ptr = static_cast<int*>(buf.ptr);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    ptr[i * n + j] = result(i, j);
                }
            }
            return output;
        } else if (method == "simd_parallel") {
            auto result = generator.generate_simd_parallel(n);
            auto output = py::array_t<int>({n, n});
            auto buf = output.request();
            int* ptr = static_cast<int*>(buf.ptr);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    ptr[i * n + j] = result(i, j);
                }
            }
            return output;
        } else {
            throw std::invalid_argument("Method must be 'blocked', 'parallel', or 'simd_parallel'");
        }
    }, py::arg("n"), py::arg("method") = "parallel", 
       "Create Hadamard matrix with optimized method");
    
    m.def("batch_fwht", [](py::array_t<double> signals) -> py::array_t<double> {
        auto buf = signals.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Signals must be 2-dimensional (batch_size, signal_length)");
        }
        
        int batch_size = buf.shape[0];
        int signal_length = buf.shape[1];
        double* data = static_cast<double*>(buf.ptr);
        
        std::vector<hatrix::simd::aligned_vector<double>> signal_vectors(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            signal_vectors[i].resize(signal_length);
            for (int j = 0; j < signal_length; ++j) {
                signal_vectors[i][j] = data[i * signal_length + j];
            }
        }
        
        hatrix::parallel::BatchProcessor processor;
        auto results = processor.process_signals_batch(signal_vectors);
        
        auto output = py::array_t<double>({batch_size, signal_length});
        auto out_buf = output.request();
        double* out_ptr = static_cast<double*>(out_buf.ptr);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < signal_length; ++j) {
                out_ptr[i * signal_length + j] = results[i][j];
            }
        }
        return output;
    }, py::arg("signals"), "Process multiple signals in parallel with optimized FWHT");
    
    m.def("get_performance_info", []() -> py::dict {
        py::dict info;
        info["avx2_available"] = hatrix::simd::g_simd_caps.avx2;
        info["avx512_available"] = hatrix::simd::g_simd_caps.avx512f;
        info["fma_available"] = hatrix::simd::g_simd_caps.fma;
        info["max_threads"] = hatrix::simd::g_simd_caps.max_threads;
        info["hardware_concurrency"] = std::thread::hardware_concurrency();
        return info;
    }, "Get system performance capabilities");
    
    //----------------------------------------------------------------------
    // CONVENIENCE FUNCTIONS
    //----------------------------------------------------------------------
    m.def("create_hadamard", [](int n, const std::string& method = "recursive") -> py::array_t<int> {
        if (method == "recursive") {
            return matrix_to_numpy(hadamard::generate_recursive(n));
        } else if (method == "iterative") {
            return matrix_to_numpy(hadamard::generate_iterative(n));
        } else if (method == "optimized") {
            // Use optimized version as default
            hatrix::parallel::ParallelHadamardGenerator generator;
            auto result = generator.generate_simd_parallel(n);
            auto output = py::array_t<int>({n, n});
            auto buf = output.request();
            int* ptr = static_cast<int*>(buf.ptr);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    ptr[i * n + j] = result(i, j);
                }
            }
            return output;
        } else {
            throw std::invalid_argument("Method must be 'recursive', 'iterative', or 'optimized'");
        }
    }, py::arg("n"), py::arg("method") = "recursive", 
       "Create Hadamard matrix with specified method");
    
    m.def("create_walsh", [](int n, const std::string& ordering = "natural") -> py::array_t<int> {
        hadamard::ordering_t order;
        if (ordering == "natural") {
            order = hadamard::ordering_t::NATURAL;
        } else if (ordering == "sequency") {
            order = hadamard::ordering_t::SEQUENCY;
        } else if (ordering == "dyadic") {
            order = hadamard::ordering_t::DYADIC;
        } else {
            throw std::invalid_argument("Ordering must be 'natural', 'sequency', or 'dyadic'");
        }
        return matrix_to_numpy(hadamard::generate_walsh(n, order));
    }, py::arg("n"), py::arg("ordering") = "natural", 
       "Create Walsh-ordered matrix with specified ordering");
    
    //----------------------------------------------------------------------
    // NUMPY COMPATIBILITY
    //----------------------------------------------------------------------
    m.def("as_numpy", [](py::array_t<int> matrix) -> py::array_t<int> {
        return matrix;  // Already a numpy array
    }, py::arg("matrix"), "Ensure array is numpy-compatible");
    
    m.def("from_numpy", [](py::array_t<int> array) -> py::array_t<int> {
        return array;  // Already a numpy array
    }, py::arg("array"), "Convert numpy array to library format");
}
