//======================================================================
// hatrix_types.hpp
//----------------------------------------------------------------------
// High-quality type safety system for the Hatrix library.
// Provides strong typing, compile-time checks, and template metaprogramming.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_TYPES_HPP
#define HATRIX_TYPES_HPP

#include <type_traits>
#include <limits>
#include <cstdint>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size_t.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/static_assert.hpp>
#include <boost/concept_check.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/string_view.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include "hatrix_exceptions.hpp"
#include "hatrix_logging.hpp"

namespace hatrix {
namespace types {

//--------------------------------------------------------------------------
// MATRIX ORDER TYPE SAFETY
//--------------------------------------------------------------------------
template<int N>
struct MatrixOrder {
    static_assert(N > 0, "Matrix order must be positive");
    static_assert((N & (N - 1)) == 0, "Matrix order must be a power of 2");
    
    static constexpr int value = N;
    static constexpr int log2_value = __builtin_ctz(N);
    static constexpr bool is_valid = true;
    
    using type = MatrixOrder<N>;
    
    constexpr operator int() const { return N; }
    constexpr int operator()() const { return N; }
};

// Invalid matrix order specialization
template<int N>
struct MatrixOrder {
    static_assert(N > 0, "Matrix order must be positive");
    static_assert((N & (N - 1)) == 0, "Matrix order must be a power of 2");
    
    static constexpr int value = N;
    static constexpr bool is_valid = false;
    
    using type = MatrixOrder<N>;
    
    constexpr operator int() const { return N; }
    constexpr int operator()() const { return N; }
};

//--------------------------------------------------------------------------
// ELEMENT TYPE CONCEPTS
//--------------------------------------------------------------------------
template<typename T>
struct is_hadamard_element : boost::mpl::bool_<
    std::is_integral<T>::value && 
    (std::is_same<T, int>::value || std::is_same<T, int8_t>::value || 
     std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value ||
     std::is_same<T, int64_t>::value)
> {};

template<typename T>
struct is_floating_element : boost::mpl::bool_<
    std::is_floating_point<T>::value
> {};

template<typename T>
struct is_numeric_element : boost::mpl::bool_<
    std::is_arithmetic<T>::value
> {};

//--------------------------------------------------------------------------
// MATRIX DIMENSION TYPES
//--------------------------------------------------------------------------
template<typename T, int Rows, int Cols>
struct MatrixDimensions {
    static_assert(Rows > 0, "Number of rows must be positive");
    static_assert(Cols > 0, "Number of columns must be positive");
    
    static constexpr int rows = Rows;
    static constexpr int cols = Cols;
    static constexpr int size = Rows * Cols;
    
    using element_type = T;
    using type = MatrixDimensions<T, Rows, Cols>;
    
    static_assert(is_numeric_element<T>::value, "Element type must be numeric");
};

// Square matrix specialization
template<typename T, int N>
struct SquareMatrixDimensions : MatrixDimensions<T, N, N> {
    static constexpr int order = N;
    static constexpr bool is_square = true;
    static constexpr bool is_power_of_two = (N & (N - 1)) == 0;
    
    static_assert(is_power_of_two, "Square matrix order must be a power of 2 for Hadamard matrices");
};

//--------------------------------------------------------------------------
// TYPE TRAITS FOR ALGORITHMS
//--------------------------------------------------------------------------
template<typename T>
struct is_simd_compatible : boost::mpl::bool_<
    std::is_same<T, float>::value || std::is_same<T, double>::value ||
    std::is_same<T, int>::value || std::is_same<T, int32_t>::value
> {};

template<typename T>
struct simd_width {
    static constexpr int value = 
        std::is_same<T, float>::value ? 8 :  // AVX2: 8 floats
        std::is_same<T, double>::value ? 4 : // AVX2: 4 doubles
        std::is_same<T, int>::value ? 8 :    // AVX2: 8 ints
        std::is_same<T, int32_t>::value ? 8 : 1;
};

template<typename T>
struct is_cache_friendly : boost::mpl::bool_<
    sizeof(T) <= 8 && std::is_trivially_copyable<T>::value
> {};

//--------------------------------------------------------------------------
// COMPILE-TIME VALIDATION
//--------------------------------------------------------------------------
template<typename T>
struct validate_hadamard_element {
    static_assert(is_hadamard_element<T>::value, 
                  "Element type must be integral for Hadamard matrices");
    static constexpr bool value = true;
};

template<int N>
struct validate_hadamard_order {
    static_assert(N > 0, "Matrix order must be positive");
    static_assert((N & (N - 1)) == 0, "Matrix order must be a power of 2");
    static constexpr bool value = true;
};

template<typename T, int N>
struct validate_hadamard_matrix {
    static_assert(validate_hadamard_element<T>::value, "Invalid element type");
    static_assert(validate_hadamard_order<N>::value, "Invalid matrix order");
    static constexpr bool value = true;
};

//--------------------------------------------------------------------------
// TYPE-SAFE MATRIX WRAPPER
//--------------------------------------------------------------------------
template<typename T, int N>
class TypedMatrix {
    static_assert(validate_hadamard_matrix<T, N>::value, "Invalid matrix type");
    
public:
    using element_type = T;
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;
    
    static constexpr int order = N;
    static constexpr int log2_order = __builtin_ctz(N);
    static constexpr size_type size = N * N;
    
    // Constructors
    constexpr TypedMatrix() : data_{} {
        HATRIX_DEBUG("Created TypedMatrix<{}, {}>", typeid(T).name(), N);
    }
    
    constexpr explicit TypedMatrix(const T& value) : data_{} {
        std::fill(data_, data_ + size, value);
        HATRIX_DEBUG("Created TypedMatrix<{}, {}> initialized with value", typeid(T).name(), N);
    }
    
    // Copy constructor
    constexpr TypedMatrix(const TypedMatrix& other) : data_{} {
        std::copy(other.data_, other.data_ + size, data_);
        HATRIX_DEBUG("Copied TypedMatrix<{}, {}>", typeid(T).name(), N);
    }
    
    // Move constructor
    constexpr TypedMatrix(TypedMatrix&& other) noexcept : data_{} {
        std::move(other.data_, other.data_ + size, data_);
        HATRIX_DEBUG("Moved TypedMatrix<{}, {}>", typeid(T).name(), N);
    }
    
    // Assignment operators
    constexpr TypedMatrix& operator=(const TypedMatrix& other) {
        if (this != &other) {
            std::copy(other.data_, other.data_ + size, data_);
        }
        return *this;
    }
    
    constexpr TypedMatrix& operator=(TypedMatrix&& other) noexcept {
        if (this != &other) {
            std::move(other.data_, other.data_ + size, data_);
        }
        return *this;
    }
    
    // Element access
    constexpr reference operator()(int row, int col) {
        HATRIX_DEBUG_ASSERT(row >= 0 && row < N, "Row index {} out of bounds [0, {})", row, N);
        HATRIX_DEBUG_ASSERT(col >= 0 && col < N, "Column index {} out of bounds [0, {})", col, N);
        return data_[row * N + col];
    }
    
    constexpr const_reference operator()(int row, int col) const {
        HATRIX_DEBUG_ASSERT(row >= 0 && row < N, "Row index {} out of bounds [0, {})", row, N);
        HATRIX_DEBUG_ASSERT(col >= 0 && col < N, "Column index {} out of bounds [0, {})", col, N);
        return data_[row * N + col];
    }
    
    constexpr reference at(int row, int col) {
        if (row < 0 || row >= N || col < 0 || col >= N) {
            HATRIX_THROW(exceptions::MathematicalError, 
                        fmt::format("Index ({}, {}) out of bounds for matrix of order {}", row, col, N));
        }
        return data_[row * N + col];
    }
    
    constexpr const_reference at(int row, int col) const {
        if (row < 0 || row >= N || col < 0 || col >= N) {
            HATRIX_THROW(exceptions::MathematicalError, 
                        fmt::format("Index ({}, {}) out of bounds for matrix of order {}", row, col, N));
        }
        return data_[row * N + col];
    }
    
    // Iterator support
    constexpr iterator begin() { return data_; }
    constexpr iterator end() { return data_ + size; }
    constexpr const_iterator begin() const { return data_; }
    constexpr const_iterator end() const { return data_; }
    constexpr const_iterator cbegin() const { return data_; }
    constexpr const_iterator cend() const { return data_; }
    
    // Data access
    constexpr pointer data() { return data_; }
    constexpr const_pointer data() const { return data_; }
    
    // Size information
    constexpr size_type size() const { return size; }
    constexpr int order() const { return N; }
    constexpr bool empty() const { return false; }
    
    // Type information
    static constexpr bool is_square() { return true; }
    static constexpr bool is_power_of_two() { return (N & (N - 1)) == 0; }
    static constexpr int log2_order() { return __builtin_ctz(N); }
    
    // Fill operations
    constexpr void fill(const T& value) {
        std::fill(data_, data_ + size, value);
    }
    
    constexpr void zero() {
        fill(T{});
    }
    
    // Swap
    constexpr void swap(TypedMatrix& other) noexcept {
        std::swap(data_, other.data_);
    }
    
private:
    T data_[N * N];
};

//--------------------------------------------------------------------------
// TYPE-SAFE VECTOR WRAPPER
//--------------------------------------------------------------------------
template<typename T, int N>
class TypedVector {
    static_assert(is_numeric_element<T>::value, "Element type must be numeric");
    static_assert(N > 0, "Vector size must be positive");
    
public:
    using element_type = T;
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;
    
    static constexpr int size = N;
    static constexpr bool is_power_of_two = (N & (N - 1)) == 0;
    
    // Constructors
    constexpr TypedVector() : data_{} {
        HATRIX_DEBUG("Created TypedVector<{}, {}>", typeid(T).name(), N);
    }
    
    constexpr explicit TypedVector(const T& value) : data_{} {
        std::fill(data_, data_ + N, value);
    }
    
    // Element access
    constexpr reference operator[](int index) {
        HATRIX_DEBUG_ASSERT(index >= 0 && index < N, "Index {} out of bounds [0, {})", index, N);
        return data_[index];
    }
    
    constexpr const_reference operator[](int index) const {
        HATRIX_DEBUG_ASSERT(index >= 0 && index < N, "Index {} out of bounds [0, {})", index, N);
        return data_[index];
    }
    
    // Iterator support
    constexpr iterator begin() { return data_; }
    constexpr iterator end() { return data_ + N; }
    constexpr const_iterator begin() const { return data_; }
    constexpr const_iterator end() const { return data_; }
    
    // Data access
    constexpr pointer data() { return data_; }
    constexpr const_pointer data() const { return data_; }
    
    // Size information
    constexpr size_type size() const { return N; }
    constexpr bool empty() const { return false; }
    
private:
    T data_[N];
};

//--------------------------------------------------------------------------
// CONVENIENCE TYPE ALIASES
//--------------------------------------------------------------------------
// Common Hadamard matrix types
template<int N>
using HadamardMatrix = TypedMatrix<int, N>;

template<int N>
using HadamardVector = TypedVector<int, N>;

// Floating point types for transforms
template<int N>
using TransformMatrix = TypedMatrix<double, N>;

template<int N>
using TransformVector = TypedVector<double, N>;

// Small matrices for optimization
template<int N>
using SmallHadamardMatrix = TypedMatrix<int8_t, N>;

//--------------------------------------------------------------------------
// TYPE UTILITIES
//--------------------------------------------------------------------------
namespace utils {
    
    // Get the appropriate type for a given size
    template<int N>
    struct size_to_type {
        using type = typename boost::mpl::if_c<
            N <= 127, int8_t,
            typename boost::mpl::if_c<
                N <= 32767, int16_t,
                typename boost::mpl::if_c<
                    N <= 2147483647, int32_t, int64_t
                >::type
            >::type
        >::type;
    };
    
    // Check if two matrices are compatible for operations
    template<typename T1, int N1, typename T2, int N2>
    struct are_compatible {
        static constexpr bool value = std::is_same<T1, T2>::value && N1 == N2;
    };
    
    // Get the result type for operations
    template<typename T1, typename T2>
    struct result_type {
        using type = decltype(T1{} + T2{});
    };
    
    // Check if a type is a valid matrix type
    template<typename T>
    struct is_matrix_type : boost::mpl::false_ {};
    
    template<typename U, int N>
    struct is_matrix_type<TypedMatrix<U, N>> : boost::mpl::true_ {};
    
    // Check if a type is a valid vector type
    template<typename T>
    struct is_vector_type : boost::mpl::false_ {};
    
    template<typename U, int N>
    struct is_vector_type<TypedVector<U, N>> : boost::mpl::true_ {};
    
} // namespace utils

//--------------------------------------------------------------------------
// CONVENIENCE FUNCTIONS
//--------------------------------------------------------------------------
template<int N>
constexpr HadamardMatrix<N> make_hadamard_matrix() {
    return HadamardMatrix<N>{};
}

template<int N>
constexpr HadamardVector<N> make_hadamard_vector() {
    return HadamardVector<N>{};
}

template<int N>
constexpr TransformMatrix<N> make_transform_matrix() {
    return TransformMatrix<N>{};
}

template<int N>
constexpr TransformVector<N> make_transform_vector() {
    return TransformVector<N>{};
}

} // namespace types
} // namespace hatrix

#endif // HATRIX_TYPES_HPP
