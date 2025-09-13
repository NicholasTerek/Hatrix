//======================================================================
// hadamard_matrix.hpp
//----------------------------------------------------------------------
// A comprehensive header-only C++17 library for generating, manipulating,
// and verifying Hadamard matrices of orders n = 2^k (Sylvester's construction).
// Designed to be impressively verbose, with extensive documentation,
// utility functions, and compile-time checks.
//
// FEATURES:
//   - Power-of-two order validation with detailed error messages.
//   - Recursive Sylvester construction and iterative expansion.
//   - Matrix transpose, multiply (matrix-vector, matrix-matrix).
//   - Orthogonality check (inner product verification).
//   - Pretty-printing with configurable formatting.
//   - Compile-time constants and versioning macros.
//   - Walsh functions generation and ordering
//   - Fast Walsh-Hadamard Transform (FWHT)
//   - Matrix serialization and deserialization
//   - Statistical analysis functions
//   - Performance benchmarking utilities
//   - Custom matrix properties validation
//   - Alternative construction methods
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HADAMARD_MATRIX_HPP
#define HADAMARD_MATRIX_HPP

#include <vector>
#include <array>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <fstream>
#include <chrono>
#include <random>
#include <cmath>
#include <unordered_map>
#include <string>
#include <functional>

// Hatrix internal headers - High-quality implementations inspired by industry leaders
#include "hatrix_exceptions.hpp"
#include "hatrix_logging.hpp"
#include "hatrix_format.hpp"
#include "hatrix_containers.hpp"
#include "hatrix_config.hpp"

namespace hadamard {

//--------------------------------------------------------------------------
// LIBRARY METADATA
//--------------------------------------------------------------------------
static constexpr const char* library_version = "1.0.0";
static constexpr const char* library_author  = "Nicholas Terek";

//--------------------------------------------------------------------------
// TYPE ALIASES
//--------------------------------------------------------------------------
// Use custom containers inspired by Boost for better performance and features
using matrix_t = std::vector<std::vector<int>>;
using vector_t = std::vector<int>;
using dvector_t = std::vector<double>;
using properties_t = std::unordered_map<std::string, double>;

// Small vector optimization for small matrices
template<std::size_t N>
using small_matrix_t = hatrix::containers::small_vector<hatrix::containers::small_vector<int, N>, N>;

// Aligned vectors for SIMD optimization
template<std::size_t Alignment = 64>
using aligned_vector_t = hatrix::containers::aligned_vector<int, Alignment>;

template<std::size_t Alignment = 64>
using aligned_dvector_t = hatrix::containers::aligned_vector<double, Alignment>;

// Optional types for safer programming (using std::optional)
using optional_matrix_t = std::optional<matrix_t>;
using optional_dvector_t = std::optional<dvector_t>;

// String view for efficient string handling
using string_view_t = std::string_view;

//--------------------------------------------------------------------------
// ENUMERATIONS
//--------------------------------------------------------------------------
enum class ordering_t {
    NATURAL,    // Natural (binary) ordering
    SEQUENCY,   // Sequency (Walsh) ordering
    DYADIC      // Dyadic ordering
};

enum class format_t {
    COMPACT,    // Compact +1/-1 format
    VERBOSE,    // Verbose with spacing
    LATEX,      // LaTeX matrix format
    CSV,        // Comma-separated values
    BINARY      // Binary 1/0 format
};

//--------------------------------------------------------------------------
// INTERNAL: Validate order
//--------------------------------------------------------------------------
inline void validate_order(int n) {
    HATRIX_DEBUG("Validating Hadamard matrix order: " + std::to_string(n));
    
    if (n <= 0) {
        HATRIX_THROW(exceptions::InvalidMatrixSize, static_cast<std::size_t>(n));
    }
    
    if ((n & (n - 1)) != 0) {
        HATRIX_THROW(exceptions::InvalidMatrixSize, static_cast<std::size_t>(n));
    }
    
    HATRIX_DEBUG("Order validation passed for size: " + std::to_string(n));
}

//--------------------------------------------------------------------------
// UTILITY: Count set bits (for Walsh ordering)
//--------------------------------------------------------------------------
inline int popcount(int x) {
    return __builtin_popcount(x);
}

//--------------------------------------------------------------------------
// UTILITY: Gray code conversion
//--------------------------------------------------------------------------
inline int binary_to_gray(int n) {
    return n ^ (n >> 1);
}

inline int gray_to_binary(int n) {
    int result = 0;
    while (n) {
        result ^= n;
        n >>= 1;
    }
    return result;
}

//--------------------------------------------------------------------------
// RECURSIVE SYLVESTER CONSTRUCTION
// H(1) = [+1]
// H(2n) = [ H(n)  H(n)
//           H(n) -H(n) ]
//--------------------------------------------------------------------------
inline matrix_t generate_recursive(int n) {
    HATRIX_PERFORMANCE_TIMER_DEBUG("generate_recursive_" + std::to_string(n));
    
    validate_order(n);
    
    if (n == 1) {
        HATRIX_DEBUG("Generating base case H(1)");
        return matrix_t{{+1}};
    }
    
    HATRIX_DEBUG("Generating H(" + std::to_string(n) + ") using recursive Sylvester construction");
    const int half = n / 2;
    matrix_t Hh = generate_recursive(half);
    matrix_t H(n, std::vector<int>(n));
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            int v = Hh[i][j];
            H[i][j]                = v;    // top-left
            H[i][j + half]         = v;    // top-right
            H[i + half][j]         = v;    // bottom-left
            H[i + half][j + half]  = -v;   // bottom-right
        }
    }
    return H;
}

//--------------------------------------------------------------------------
// ITERATIVE EXPANSION (ALTERNATIVE)
//--------------------------------------------------------------------------
inline matrix_t generate_iterative(int n) {
    validate_order(n);
    matrix_t H = {{+1}};
    for (int size = 1; size < n; size <<= 1) {
        matrix_t next(2 * size, std::vector<int>(2 * size));
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                int v = H[i][j];
                next[i][j]             = v;
                next[i][j + size]      = v;
                next[i + size][j]      = v;
                next[i + size][j + size] = -v;
            }
        }
        H.swap(next);
    }
    return H;
}

//--------------------------------------------------------------------------
// WALSH FUNCTIONS GENERATION
//--------------------------------------------------------------------------
inline matrix_t generate_walsh(int n, ordering_t order = ordering_t::NATURAL) {
    validate_order(n);
    matrix_t H = generate_recursive(n);
    
    if (order == ordering_t::NATURAL) {
        return H;
    }
    
    matrix_t W(n, std::vector<int>(n));
    std::vector<int> row_order(n);
    
    // Generate ordering indices
    for (int i = 0; i < n; ++i) {
        row_order[i] = i;
    }
    
    if (order == ordering_t::SEQUENCY) {
        // Sort by number of sign changes (sequency)
        std::sort(row_order.begin(), row_order.end(), [&H, n](int a, int b) {
            auto count_changes = [&H, n](int row) {
                int changes = 0;
                for (int j = 1; j < n; ++j) {
                    if (H[row][j] != H[row][j-1]) changes++;
                }
                return changes;
            };
            return count_changes(a) < count_changes(b);
        });
    } else if (order == ordering_t::DYADIC) {
        // Use Gray code ordering
        for (int i = 0; i < n; ++i) {
            row_order[i] = binary_to_gray(i);
        }
    }
    
    // Reorder rows
    for (int i = 0; i < n; ++i) {
        W[i] = H[row_order[i]];
    }
    
    return W;
}

//--------------------------------------------------------------------------
// FAST WALSH-HADAMARD TRANSFORM (FWHT)
//--------------------------------------------------------------------------
inline dvector_t fwht(dvector_t data) {
    int n = static_cast<int>(data.size());
    validate_order(n);
    
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                double u = data[i + j];
                double v = data[i + j + len / 2];
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
            }
        }
    }
    
    // Normalize
    double norm = 1.0 / std::sqrt(n);
    for (auto& val : data) {
        val *= norm;
    }
    
    return data;
}

//--------------------------------------------------------------------------
// INVERSE FAST WALSH-HADAMARD TRANSFORM
//--------------------------------------------------------------------------
inline dvector_t ifwht(dvector_t data) {
    // FWHT is its own inverse (up to scaling)
    return fwht(data);
}

//--------------------------------------------------------------------------
// TRANSPOSE
//--------------------------------------------------------------------------
inline matrix_t transpose(const matrix_t& M) {
    int n = static_cast<int>(M.size());
    if (n == 0) return {};
    matrix_t T(n, std::vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            T[j][i] = M[i][j];
    return T;
}

//--------------------------------------------------------------------------
// MATRIX-VECTOR MULTIPLICATION
//--------------------------------------------------------------------------
inline vector_t multiply(const matrix_t& M, const vector_t& v) {
    int n = static_cast<int>(M.size());
    if (static_cast<int>(v.size()) != n) {
        throw std::invalid_argument("[Hadamard] Dimension mismatch in multiply");
    }
    vector_t out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = std::inner_product(M[i].begin(), M[i].end(), v.begin(), 0);
    }
    return out;
}

//--------------------------------------------------------------------------
// MATRIX-MATRIX MULTIPLICATION
//--------------------------------------------------------------------------
inline matrix_t multiply(const matrix_t& A, const matrix_t& B) {
    int n = static_cast<int>(A.size());
    if (static_cast<int>(B.size()) != n) {
        throw std::invalid_argument("[Hadamard] Dimension mismatch in multiply");
    }
    matrix_t C(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            int a = A[i][k];
            for (int j = 0; j < n; ++j) {
                C[i][j] += a * B[k][j];
            }
        }
    }
    return C;
}

//--------------------------------------------------------------------------
// ORTHOGONALITY CHECK
// Verifies H * H^T == n * I_n
//--------------------------------------------------------------------------
inline bool is_orthogonal(const matrix_t& H) {
    int n = static_cast<int>(H.size());
    auto HT = transpose(H);
    auto P  = multiply(H, HT);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int expected = (i == j ? n : 0);
            if (P[i][j] != expected) return false;
        }
    }
    return true;
}

//--------------------------------------------------------------------------
// HADAMARD PROPERTY VALIDATION
//--------------------------------------------------------------------------
inline bool is_hadamard(const matrix_t& H) {
    int n = static_cast<int>(H.size());
    if (n == 0) return false;
    
    // Check if all entries are ±1
    for (const auto& row : H) {
        if (static_cast<int>(row.size()) != n) return false;
        for (int val : row) {
            if (val != 1 && val != -1) return false;
        }
    }
    
    // Check orthogonality
    return is_orthogonal(H);
}

//--------------------------------------------------------------------------
// MATRIX PROPERTIES ANALYSIS
//--------------------------------------------------------------------------
inline properties_t analyze_properties(const matrix_t& H) {
    properties_t props;
    int n = static_cast<int>(H.size());
    
    props["size"] = n;
    props["is_hadamard"] = is_hadamard(H) ? 1.0 : 0.0;
    props["is_orthogonal"] = is_orthogonal(H) ? 1.0 : 0.0;
    
    // Calculate determinant sign (should be ±1 for Hadamard)
    // For power-of-2 Hadamard matrices, |det| = n^(n/2)
    double expected_det_magnitude = std::pow(n, n / 2.0);
    props["expected_det_magnitude"] = expected_det_magnitude;
    
    // Count sign changes in each row (sequency)
    double avg_sequency = 0.0;
    for (const auto& row : H) {
        int changes = 0;
        for (int j = 1; j < n; ++j) {
            if (row[j] != row[j-1]) changes++;
        }
        avg_sequency += changes;
    }
    props["average_sequency"] = avg_sequency / n;
    
    // Calculate condition number (should be sqrt(n) for Hadamard)
    props["expected_condition_number"] = std::sqrt(n);
    
    return props;
}

//--------------------------------------------------------------------------
// MATRIX SERIALIZATION
//--------------------------------------------------------------------------
inline std::string serialize(const matrix_t& H, format_t fmt = format_t::COMPACT) {
    std::ostringstream oss;
    int n = static_cast<int>(H.size());
    
    switch (fmt) {
        case format_t::COMPACT:
            oss << n << "\n";
            for (const auto& row : H) {
                for (int val : row) {
                    oss << (val > 0 ? '+' : '-');
                }
                oss << "\n";
            }
            break;
            
        case format_t::CSV:
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    oss << H[i][j];
                    if (j < n - 1) oss << ",";
                }
                oss << "\n";
            }
            break;
            
        case format_t::LATEX:
            oss << "\\begin{pmatrix}\n";
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    oss << std::setw(2) << H[i][j];
                    if (j < n - 1) oss << " & ";
                }
                oss << " \\\\\n";
            }
            oss << "\\end{pmatrix}";
            break;
            
        case format_t::BINARY:
            oss << n << "\n";
            for (const auto& row : H) {
                for (int val : row) {
                    oss << (val > 0 ? '1' : '0');
                }
                oss << "\n";
            }
            break;
            
        default:
            // VERBOSE format
            oss << "Hadamard Matrix " << n << "x" << n << "\n";
            for (const auto& row : H) {
                oss << "[ ";
                for (int val : row) {
                    oss << std::setw(3) << (val > 0 ? "+1" : "-1");
                }
                oss << " ]\n";
            }
            break;
    }
    
    return oss.str();
}

//--------------------------------------------------------------------------
// MATRIX DESERIALIZATION
//--------------------------------------------------------------------------
inline matrix_t deserialize(const std::string& data, format_t fmt = format_t::COMPACT) {
    std::istringstream iss(data);
    matrix_t H;
    
    if (fmt == format_t::COMPACT || fmt == format_t::BINARY) {
        int n;
        iss >> n;
        validate_order(n);
        
        H.resize(n, std::vector<int>(n));
        std::string line;
        std::getline(iss, line); // consume newline
        
        for (int i = 0; i < n; ++i) {
            std::getline(iss, line);
            if (line.length() != static_cast<size_t>(n)) {
                throw std::invalid_argument("[Hadamard] Invalid serialized format");
            }
            for (int j = 0; j < n; ++j) {
                if (fmt == format_t::COMPACT) {
                    H[i][j] = (line[j] == '+') ? 1 : -1;
                } else { // BINARY
                    H[i][j] = (line[j] == '1') ? 1 : -1;
                }
            }
        }
    } else if (fmt == format_t::CSV) {
        std::string line;
        while (std::getline(iss, line)) {
            std::vector<int> row;
            std::istringstream line_stream(line);
            std::string cell;
            
            while (std::getline(line_stream, cell, ',')) {
                row.push_back(std::stoi(cell));
            }
            
            if (!row.empty()) {
                H.push_back(row);
            }
        }
    }
    
    return H;
}

//--------------------------------------------------------------------------
// FILE I/O OPERATIONS
//--------------------------------------------------------------------------
inline void save_to_file(const matrix_t& H, const std::string& filename, format_t fmt = format_t::COMPACT) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("[Hadamard] Cannot open file for writing: " + filename);
    }
    file << serialize(H, fmt);
    file.close();
}

inline matrix_t load_from_file(const std::string& filename, format_t fmt = format_t::COMPACT) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("[Hadamard] Cannot open file for reading: " + filename);
    }
    
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    return deserialize(content, fmt);
}

//--------------------------------------------------------------------------
// PERFORMANCE BENCHMARKING
//--------------------------------------------------------------------------
template<typename Func>
inline double benchmark(Func&& func, int iterations = 1) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / static_cast<double>(iterations);
}

//--------------------------------------------------------------------------
// RANDOM HADAMARD-LIKE MATRIX GENERATION (FOR TESTING)
//--------------------------------------------------------------------------
inline matrix_t generate_random_binary(int n, int seed = 42) {
    validate_order(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 1);
    
    matrix_t M(n, std::vector<int>(n));
    for (auto& row : M) {
        for (auto& val : row) {
            val = dis(gen) ? 1 : -1;
        }
    }
    return M;
}

//--------------------------------------------------------------------------
// PRETTY-PRINT WITH ENHANCED FORMATTING
//--------------------------------------------------------------------------
inline void print(const matrix_t& H, std::ostream& os = std::cout, format_t fmt = format_t::VERBOSE) {
    int n = static_cast<int>(H.size());
    
    switch (fmt) {
        case format_t::COMPACT:
            for (const auto& row : H) {
                for (int val : row) {
                    os << (val > 0 ? '+' : '-');
                }
                os << "\n";
            }
            break;
            
        case format_t::LATEX:
            os << serialize(H, format_t::LATEX);
            break;
            
        case format_t::CSV:
            os << serialize(H, format_t::CSV);
            break;
            
        case format_t::BINARY:
            for (const auto& row : H) {
                for (int val : row) {
                    os << (val > 0 ? '1' : '0') << " ";
                }
                os << "\n";
            }
            break;
            
        default: // VERBOSE
            os << "// Hadamard Matrix (" << n << "x" << n << ") v" << library_version << "\n";
            for (int i = 0; i < n; ++i) {
                os << "[ ";
                for (int j = 0; j < n; ++j) {
                    os << std::setw(2) << (H[i][j] > 0 ? "+1" : "-1");
                    if (j + 1 < n) os << ",";
                    os << " ";
                }
                os << "]\n";
            }
            break;
    }
}

//--------------------------------------------------------------------------
// COMPREHENSIVE MATRIX VALIDATION
//--------------------------------------------------------------------------
inline std::vector<std::string> validate_matrix(const matrix_t& H) {
    std::vector<std::string> issues;
    int n = static_cast<int>(H.size());
    
    // Check if empty
    if (n == 0) {
        issues.push_back("Matrix is empty");
        return issues;
    }
    
    // Check if square
    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(H[i].size()) != n) {
            issues.push_back("Matrix is not square at row " + std::to_string(i));
        }
    }
    
    // Check if power of 2
    if ((n & (n - 1)) != 0) {
        issues.push_back("Matrix size is not a power of 2");
    }
    
    // Check if entries are ±1
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (H[i][j] != 1 && H[i][j] != -1) {
                issues.push_back("Invalid entry at (" + std::to_string(i) + "," + std::to_string(j) + "): " + std::to_string(H[i][j]));
            }
        }
    }
    
    // Check orthogonality
    if (!is_orthogonal(H)) {
        issues.push_back("Matrix is not orthogonal");
    }
    
    return issues;
}

} // namespace hadamard

#endif // HADAMARD_MATRIX_HPP
