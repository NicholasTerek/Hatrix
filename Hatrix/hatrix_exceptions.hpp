//======================================================================
// hatrix_exceptions.hpp
//----------------------------------------------------------------------
// High-quality exception hierarchy for the Hatrix library.
// Provides comprehensive error handling with detailed error information.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_EXCEPTIONS_HPP
#define HATRIX_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>
#include <sstream>
#include <boost/exception/all.hpp>
#include <boost/stacktrace.hpp>
#include <fmt/format.h>

namespace hatrix {
namespace exceptions {

//--------------------------------------------------------------------------
// EXCEPTION TAGS FOR BOOST.Exception
//--------------------------------------------------------------------------
using error_info_tag = boost::error_info<struct tag_error_info, std::string>;
using matrix_size_tag = boost::error_info<struct tag_matrix_size, std::size_t>;
using operation_tag = boost::error_info<struct tag_operation, std::string>;
using file_path_tag = boost::error_info<struct tag_file_path, std::string>;
using line_number_tag = boost::error_info<struct tag_line_number, int>;
using stacktrace_tag = boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace>;

//--------------------------------------------------------------------------
// BASE HATRIX EXCEPTION
//--------------------------------------------------------------------------
class HatrixException : public std::runtime_error, public boost::exception {
public:
    explicit HatrixException(const std::string& message) 
        : std::runtime_error(message) {}
    
    explicit HatrixException(const char* message) 
        : std::runtime_error(message) {}
    
    virtual ~HatrixException() noexcept = default;
    
    // Add stack trace automatically
    HatrixException() {
        *this << stacktrace_tag(boost::stacktrace::stacktrace());
    }
    
    HatrixException(const std::string& message, const std::string& operation) 
        : std::runtime_error(message) {
        *this << operation_tag(operation);
        *this << stacktrace_tag(boost::stacktrace::stacktrace());
    }
};

//--------------------------------------------------------------------------
// MATHEMATICAL EXCEPTIONS
//--------------------------------------------------------------------------
class MathematicalError : public HatrixException {
public:
    explicit MathematicalError(const std::string& message) 
        : HatrixException(message, "mathematical_operation") {}
};

class InvalidMatrixSize : public MathematicalError {
public:
    explicit InvalidMatrixSize(std::size_t size) 
        : MathematicalError(fmt::format("Invalid matrix size: {} (must be power of 2)", size)) {
        *this << matrix_size_tag(size);
    }
};

class DimensionMismatch : public MathematicalError {
public:
    DimensionMismatch(std::size_t expected, std::size_t actual, const std::string& operation = "")
        : MathematicalError(fmt::format("Dimension mismatch: expected {}, got {}", expected, actual)) {
        *this << matrix_size_tag(actual);
        if (!operation.empty()) {
            *this << operation_tag(operation);
        }
    }
};

class SingularMatrix : public MathematicalError {
public:
    explicit SingularMatrix(const std::string& operation = "matrix_inverse")
        : MathematicalError("Matrix is singular (non-invertible)") {
        *this << operation_tag(operation);
    }
};

//--------------------------------------------------------------------------
// PERFORMANCE EXCEPTIONS
//--------------------------------------------------------------------------
class PerformanceError : public HatrixException {
public:
    explicit PerformanceError(const std::string& message) 
        : HatrixException(message, "performance_operation") {}
};

class SIMDNotSupported : public PerformanceError {
public:
    explicit SIMDNotSupported(const std::string& feature)
        : PerformanceError(fmt::format("SIMD feature not supported: {}", feature)) {
        *this << error_info_tag(feature);
    }
};

class MemoryAllocationError : public PerformanceError {
public:
    explicit MemoryAllocationError(std::size_t size)
        : PerformanceError(fmt::format("Failed to allocate {} bytes", size)) {
        *this << matrix_size_tag(size);
    }
};

//--------------------------------------------------------------------------
// I/O EXCEPTIONS
//--------------------------------------------------------------------------
class IOError : public HatrixException {
public:
    explicit IOError(const std::string& message) 
        : HatrixException(message, "io_operation") {}
};

class FileNotFound : public IOError {
public:
    explicit FileNotFound(const std::string& filepath)
        : IOError(fmt::format("File not found: {}", filepath)) {
        *this << file_path_tag(filepath);
    }
};

class FileFormatError : public IOError {
public:
    FileFormatError(const std::string& filepath, const std::string& reason)
        : IOError(fmt::format("File format error in {}: {}", filepath, reason)) {
        *this << file_path_tag(filepath);
        *this << error_info_tag(reason);
    }
};

class SerializationError : public IOError {
public:
    SerializationError(const std::string& operation, const std::string& reason)
        : IOError(fmt::format("Serialization error during {}: {}", operation, reason)) {
        *this << operation_tag(operation);
        *this << error_info_tag(reason);
    }
};

//--------------------------------------------------------------------------
// THREADING EXCEPTIONS
//--------------------------------------------------------------------------
class ThreadingError : public HatrixException {
public:
    explicit ThreadingError(const std::string& message) 
        : HatrixException(message, "threading_operation") {}
};

class ThreadPoolError : public ThreadingError {
public:
    explicit ThreadPoolError(const std::string& reason)
        : ThreadingError(fmt::format("Thread pool error: {}", reason)) {
        *this << error_info_tag(reason);
    }
};

class DeadlockDetected : public ThreadingError {
public:
    explicit DeadlockDetected(const std::string& operation)
        : ThreadingError(fmt::format("Deadlock detected in operation: {}", operation)) {
        *this << operation_tag(operation);
    }
};

//--------------------------------------------------------------------------
// VALIDATION EXCEPTIONS
//--------------------------------------------------------------------------
class ValidationError : public HatrixException {
public:
    explicit ValidationError(const std::string& message) 
        : HatrixException(message, "validation") {}
};

class InvalidHadamardMatrix : public ValidationError {
public:
    InvalidHadamardMatrix(const std::string& reason, std::size_t size = 0)
        : ValidationError(fmt::format("Invalid Hadamard matrix: {}", reason)) {
        if (size > 0) {
            *this << matrix_size_tag(size);
        }
        *this << error_info_tag(reason);
    }
};

class OrthogonalityViolation : public ValidationError {
public:
    OrthogonalityViolation(std::size_t row1, std::size_t row2)
        : ValidationError(fmt::format("Orthogonality violation between rows {} and {}", row1, row2)) {
        *this << error_info_tag(fmt::format("rows_{}_and_{}", row1, row2));
    }
};

//--------------------------------------------------------------------------
// CONFIGURATION EXCEPTIONS
//--------------------------------------------------------------------------
class ConfigurationError : public HatrixException {
public:
    explicit ConfigurationError(const std::string& message) 
        : HatrixException(message, "configuration") {}
};

class InvalidParameter : public ConfigurationError {
public:
    InvalidParameter(const std::string& parameter, const std::string& value, const std::string& reason)
        : ConfigurationError(fmt::format("Invalid parameter {} = {}: {}", parameter, value, reason)) {
        *this << error_info_tag(fmt::format("{}={}", parameter, value));
    }
};

//--------------------------------------------------------------------------
// UTILITY FUNCTIONS
//--------------------------------------------------------------------------
namespace detail {
    template<typename Exception, typename... Args>
    [[noreturn]] void throw_with_context(const std::string& file, int line, Args&&... args) {
        throw Exception(std::forward<Args>(args)...) << line_number_tag(line) 
                                                    << file_path_tag(file)
                                                    << stacktrace_tag(boost::stacktrace::stacktrace());
    }
}

// Macro for throwing exceptions with file and line information
#define HATRIX_THROW(Exception, ...) \
    hatrix::exceptions::detail::throw_with_context<Exception>(__FILE__, __LINE__, ##__VA_ARGS__)

// Macro for throwing with operation context
#define HATRIX_THROW_OP(Exception, operation, ...) \
    do { \
        auto ex = Exception(__VA_ARGS__); \
        ex << hatrix::exceptions::operation_tag(operation); \
        throw ex; \
    } while(0)

//--------------------------------------------------------------------------
// EXCEPTION SAFETY GUARANTEES
//--------------------------------------------------------------------------
namespace safety {
    // Basic guarantee: No resource leaks, objects in valid state
    constexpr const char* basic_guarantee = "basic";
    
    // Strong guarantee: Operation succeeds completely or fails completely
    constexpr const char* strong_guarantee = "strong";
    
    // No-throw guarantee: Operation never throws
    constexpr const char* no_throw_guarantee = "no-throw";
}

//--------------------------------------------------------------------------
// ERROR CODE MAPPING
//--------------------------------------------------------------------------
enum class ErrorCode : int {
    Success = 0,
    InvalidMatrixSize = 1,
    DimensionMismatch = 2,
    SingularMatrix = 3,
    SIMDNotSupported = 4,
    MemoryAllocationError = 5,
    FileNotFound = 6,
    FileFormatError = 7,
    SerializationError = 8,
    ThreadPoolError = 9,
    DeadlockDetected = 10,
    InvalidHadamardMatrix = 11,
    OrthogonalityViolation = 12,
    InvalidParameter = 13,
    UnknownError = 99
};

inline const char* to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success: return "Success";
        case ErrorCode::InvalidMatrixSize: return "Invalid matrix size";
        case ErrorCode::DimensionMismatch: return "Dimension mismatch";
        case ErrorCode::SingularMatrix: return "Singular matrix";
        case ErrorCode::SIMDNotSupported: return "SIMD not supported";
        case ErrorCode::MemoryAllocationError: return "Memory allocation error";
        case ErrorCode::FileNotFound: return "File not found";
        case ErrorCode::FileFormatError: return "File format error";
        case ErrorCode::SerializationError: return "Serialization error";
        case ErrorCode::ThreadPoolError: return "Thread pool error";
        case ErrorCode::DeadlockDetected: return "Deadlock detected";
        case ErrorCode::InvalidHadamardMatrix: return "Invalid Hadamard matrix";
        case ErrorCode::OrthogonalityViolation: return "Orthogonality violation";
        case ErrorCode::InvalidParameter: return "Invalid parameter";
        case ErrorCode::UnknownError: return "Unknown error";
        default: return "Unknown error code";
    }
}

} // namespace exceptions
} // namespace hatrix

#endif // HATRIX_EXCEPTIONS_HPP
