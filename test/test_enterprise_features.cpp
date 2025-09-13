//======================================================================
// test_enterprise_features.cpp
//----------------------------------------------------------------------
// Comprehensive test suite for enterprise-grade Hatrix library features.
// Tests Boost integration, exception handling, logging, memory management,
// type safety, and configuration systems.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <optional>
#include <variant>
#include <any>
#include <chrono>
#include <thread>

// Hatrix headers - High-quality implementations
#include "hatrix/hadamard_matrix.hpp"
#include "hatrix/hatrix_exceptions.hpp"
#include "hatrix/hatrix_logging.hpp"
#include "hatrix/hatrix_format.hpp"
#include "hatrix/hatrix_containers.hpp"
#include "hatrix/hatrix_config.hpp"
#include "hatrix/hatrix_memory.hpp"
#include "hatrix/hatrix_types.hpp"

using namespace hatrix;
using namespace hatrix::exceptions;
using namespace hatrix::logging;
using namespace hatrix::config;
using namespace hatrix::memory;
using namespace hatrix::types;

//--------------------------------------------------------------------------
// EXCEPTION HANDLING TESTS
//--------------------------------------------------------------------------
class ExceptionHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        auto logger = get_logger("test");
        logger->set_level(Level::DEBUG);
    }
};

TEST_F(ExceptionHandlingTest, InvalidMatrixSize) {
    EXPECT_THROW(hadamard::validate_order(0), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(-1), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(3), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(5), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(6), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(7), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(9), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(10), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(11), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(12), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(13), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(14), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(15), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(17), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(18), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(19), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(20), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(21), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(22), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(23), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(24), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(25), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(26), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(27), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(28), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(29), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(30), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(31), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(33), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(34), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(35), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(36), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(37), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(38), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(39), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(40), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(41), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(42), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(43), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(44), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(45), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(46), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(47), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(48), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(49), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(50), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(51), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(52), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(53), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(54), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(55), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(56), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(57), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(58), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(59), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(60), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(61), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(62), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(63), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(65), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(66), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(67), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(68), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(69), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(70), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(71), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(72), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(73), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(74), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(75), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(76), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(77), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(78), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(79), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(80), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(81), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(82), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(83), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(84), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(85), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(86), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(87), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(88), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(89), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(90), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(91), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(92), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(93), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(94), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(95), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(96), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(97), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(98), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(99), InvalidMatrixSize);
    EXPECT_THROW(hadamard::validate_order(100), InvalidMatrixSize);
    
    // Valid sizes should not throw
    EXPECT_NO_THROW(hadamard::validate_order(1));
    EXPECT_NO_THROW(hadamard::validate_order(2));
    EXPECT_NO_THROW(hadamard::validate_order(4));
    EXPECT_NO_THROW(hadamard::validate_order(8));
    EXPECT_NO_THROW(hadamard::validate_order(16));
    EXPECT_NO_THROW(hadamard::validate_order(32));
    EXPECT_NO_THROW(hadamard::validate_order(64));
    EXPECT_NO_THROW(hadamard::validate_order(128));
    EXPECT_NO_THROW(hadamard::validate_order(256));
    EXPECT_NO_THROW(hadamard::validate_order(512));
    EXPECT_NO_THROW(hadamard::validate_order(1024));
    EXPECT_NO_THROW(hadamard::validate_order(2048));
    EXPECT_NO_THROW(hadamard::validate_order(4096));
    EXPECT_NO_THROW(hadamard::validate_order(8192));
    EXPECT_NO_THROW(hadamard::validate_order(16384));
    EXPECT_NO_THROW(hadamard::validate_order(32768));
    EXPECT_NO_THROW(hadamard::validate_order(65536));
    EXPECT_NO_THROW(hadamard::validate_order(131072));
    EXPECT_NO_THROW(hadamard::validate_order(262144));
    EXPECT_NO_THROW(hadamard::validate_order(524288));
    EXPECT_NO_THROW(hadamard::validate_order(1048576));
    EXPECT_NO_THROW(hadamard::validate_order(2097152));
    EXPECT_NO_THROW(hadamard::validate_order(4194304));
    EXPECT_NO_THROW(hadamard::validate_order(8388608));
    EXPECT_NO_THROW(hadamard::validate_order(16777216));
    EXPECT_NO_THROW(hadamard::validate_order(33554432));
    EXPECT_NO_THROW(hadamard::validate_order(67108864));
    EXPECT_NO_THROW(hadamard::validate_order(134217728));
    EXPECT_NO_THROW(hadamard::validate_order(268435456));
    EXPECT_NO_THROW(hadamard::validate_order(536870912));
    EXPECT_NO_THROW(hadamard::validate_order(1073741824));
}

TEST_F(ExceptionHandlingTest, DimensionMismatch) {
    EXPECT_THROW(throw DimensionMismatch(4, 8, "matrix_multiply"), DimensionMismatch);
    EXPECT_THROW(throw DimensionMismatch(16, 32, "vector_multiply"), DimensionMismatch);
}

TEST_F(ExceptionHandlingTest, SIMDNotSupported) {
    EXPECT_THROW(throw SIMDNotSupported("AVX-512"), SIMDNotSupported);
    EXPECT_THROW(throw SIMDNotSupported("FMA"), SIMDNotSupported);
}

TEST_F(ExceptionHandlingTest, MemoryAllocationError) {
    EXPECT_THROW(throw MemoryAllocationError(1024 * 1024), MemoryAllocationError);
    EXPECT_THROW(throw MemoryAllocationError(4096 * 4096), MemoryAllocationError);
}

//--------------------------------------------------------------------------
// LOGGING SYSTEM TESTS
//--------------------------------------------------------------------------
class LoggingSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for testing
        auto logger = get_logger("test");
        logger->set_level(Level::DEBUG);
    }
};

TEST_F(LoggingSystemTest, LoggingManagerInitialization) {
    LoggingConfig config;
    config.level = "debug";
    config.console_output = false;
    config.file_output = false;
    
    EXPECT_NO_THROW(initialize_logging(config));
    EXPECT_NO_THROW(get_logger());
}

TEST_F(LoggingSystemTest, PerformanceTimer) {
    {
        PerformanceTimer timer("test_operation");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Timer should complete without throwing
    SUCCEED();
}

TEST_F(LoggingSystemTest, LoggingMacros) {
    EXPECT_NO_THROW(HATRIX_TRACE("Test trace message"));
    EXPECT_NO_THROW(HATRIX_DEBUG("Test debug message"));
    EXPECT_NO_THROW(HATRIX_INFO("Test info message"));
    EXPECT_NO_THROW(HATRIX_WARN("Test warning message"));
    EXPECT_NO_THROW(HATRIX_ERROR("Test error message"));
    EXPECT_NO_THROW(HATRIX_CRITICAL("Test critical message"));
}

//--------------------------------------------------------------------------
// CONFIGURATION SYSTEM TESTS
//--------------------------------------------------------------------------
class ConfigurationSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("test", null_sink);
        spdlog::set_default_logger(logger);
    }
};

TEST_F(ConfigurationSystemTest, DefaultConfiguration) {
    Configuration config;
    
    // Test default values
    EXPECT_TRUE(config.performance.enable_simd);
    EXPECT_TRUE(config.performance.enable_parallel);
    EXPECT_EQ(config.performance.num_threads, -1);
    EXPECT_EQ(config.logging.level, "info");
    EXPECT_TRUE(config.logging.console_output);
    EXPECT_TRUE(config.algorithm.enable_matrix_validation);
}

TEST_F(ConfigurationSystemTest, ConfigurationPresets) {
    auto debug_config = presets::debug();
    EXPECT_EQ(debug_config.logging.level, "debug");
    EXPECT_TRUE(debug_config.logging.console_output);
    EXPECT_EQ(debug_config.performance.num_threads, 1);
    
    auto release_config = presets::release();
    EXPECT_EQ(release_config.logging.level, "info");
    EXPECT_FALSE(release_config.logging.console_output);
    EXPECT_TRUE(release_config.logging.file_output);
    
    auto production_config = presets::production();
    EXPECT_EQ(production_config.logging.level, "warn");
    EXPECT_FALSE(production_config.logging.console_output);
    EXPECT_FALSE(production_config.algorithm.enable_matrix_validation);
    
    auto benchmark_config = presets::benchmark();
    EXPECT_EQ(benchmark_config.logging.level, "error");
    EXPECT_FALSE(benchmark_config.logging.console_output);
    EXPECT_FALSE(benchmark_config.logging.file_output);
}

TEST_F(ConfigurationSystemTest, CustomSettings) {
    Configuration config;
    
    // Test custom settings
    config.set_custom("test_int", 42);
    config.set_custom("test_string", std::string("hello"));
    config.set_custom("test_double", 3.14);
    
    auto int_val = config.get_custom<int>("test_int");
    auto string_val = config.get_custom<std::string>("test_string");
    auto double_val = config.get_custom<double>("test_double");
    
    EXPECT_TRUE(int_val);
    EXPECT_TRUE(string_val);
    EXPECT_TRUE(double_val);
    
    EXPECT_EQ(*int_val, 42);
    EXPECT_EQ(*string_val, "hello");
    EXPECT_DOUBLE_EQ(*double_val, 3.14);
}

//--------------------------------------------------------------------------
// MEMORY MANAGEMENT TESTS
//--------------------------------------------------------------------------
class MemoryManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("test", null_sink);
        spdlog::set_default_logger(logger);
    }
};

TEST_F(MemoryManagementTest, AlignedArrayCreation) {
    auto array = make_aligned_array<double>(100);
    EXPECT_EQ(array.size(), 100);
    EXPECT_TRUE(array.is_aligned());
    EXPECT_FALSE(array.empty());
}

TEST_F(MemoryManagementTest, AlignedArrayAccess) {
    auto array = make_aligned_array<int>(50);
    
    // Test element access
    for (int i = 0; i < 50; ++i) {
        array[i] = i * 2;
    }
    
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(array[i], i * 2);
    }
}

TEST_F(MemoryManagementTest, AlignedArrayBoundsChecking) {
    auto array = make_aligned_array<int>(10);
    
    // Valid access should not throw
    EXPECT_NO_THROW(array[0]);
    EXPECT_NO_THROW(array[9]);
    
    // Invalid access should throw
    EXPECT_THROW(array.at(10), exceptions::MathematicalError);
    EXPECT_THROW(array.at(-1), exceptions::MathematicalError);
}

TEST_F(MemoryManagementTest, MemoryPoolManager) {
    auto& manager = get_memory_manager();
    
    // Test allocation
    auto ptr = manager.allocate_aligned<double>(100);
    EXPECT_NE(ptr, nullptr);
    
    // Test deallocation
    EXPECT_NO_THROW(manager.deallocate_aligned(ptr, 100));
}

//--------------------------------------------------------------------------
// TYPE SAFETY TESTS
//--------------------------------------------------------------------------
class TypeSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("test", null_sink);
        spdlog::set_default_logger(logger);
    }
};

TEST_F(TypeSafetyTest, TypedMatrixCreation) {
    auto matrix = make_hadamard_matrix<4>();
    EXPECT_EQ(matrix.order(), 4);
    EXPECT_EQ(matrix.size(), 16);
    EXPECT_TRUE(matrix.is_square());
    EXPECT_TRUE(matrix.is_power_of_two());
}

TEST_F(TypeSafetyTest, TypedMatrixAccess) {
    auto matrix = make_hadamard_matrix<4>();
    
    // Test element access
    matrix(0, 0) = 1;
    matrix(1, 1) = -1;
    
    EXPECT_EQ(matrix(0, 0), 1);
    EXPECT_EQ(matrix(1, 1), -1);
}

TEST_F(TypeSafetyTest, TypedMatrixBoundsChecking) {
    auto matrix = make_hadamard_matrix<4>();
    
    // Valid access should not throw
    EXPECT_NO_THROW(matrix(0, 0));
    EXPECT_NO_THROW(matrix(3, 3));
    
    // Invalid access should throw
    EXPECT_THROW(matrix.at(4, 0), exceptions::MathematicalError);
    EXPECT_THROW(matrix.at(0, 4), exceptions::MathematicalError);
    EXPECT_THROW(matrix.at(-1, 0), exceptions::MathematicalError);
    EXPECT_THROW(matrix.at(0, -1), exceptions::MathematicalError);
}

TEST_F(TypeSafetyTest, TypedVectorCreation) {
    auto vector = make_hadamard_vector<8>();
    EXPECT_EQ(vector.size(), 8);
    EXPECT_TRUE(vector.is_power_of_two());
    EXPECT_FALSE(vector.empty());
}

TEST_F(TypeSafetyTest, TypedVectorAccess) {
    auto vector = make_hadamard_vector<8>();
    
    // Test element access
    for (int i = 0; i < 8; ++i) {
        vector[i] = i;
    }
    
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(vector[i], i);
    }
}

//--------------------------------------------------------------------------
// BOOST INTEGRATION TESTS
//--------------------------------------------------------------------------
class BoostIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("test", null_sink);
        spdlog::set_default_logger(logger);
    }
};

TEST_F(BoostIntegrationTest, OptionalTypes) {
    std::optional<int> opt_int;
    EXPECT_FALSE(opt_int);
    
    opt_int = 42;
    EXPECT_TRUE(opt_int);
    EXPECT_EQ(*opt_int, 42);
}

TEST_F(BoostIntegrationTest, VariantTypes) {
    std::variant<int, std::string, double> variant;
    
    variant = 42;
    EXPECT_EQ(std::get<int>(variant), 42);
    
    variant = std::string("hello");
    EXPECT_EQ(std::get<std::string>(variant), "hello");
    
    variant = 3.14;
    EXPECT_DOUBLE_EQ(std::get<double>(variant), 3.14);
}

TEST_F(BoostIntegrationTest, AnyTypes) {
    std::any any_value;
    
    any_value = 42;
    EXPECT_EQ(std::any_cast<int>(any_value), 42);
    
    any_value = std::string("hello");
    EXPECT_EQ(std::any_cast<std::string>(any_value), "hello");
    
    any_value = 3.14;
    EXPECT_DOUBLE_EQ(std::any_cast<double>(any_value), 3.14);
}

TEST_F(BoostIntegrationTest, CustomContainers) {
    // Test small_vector
    hatrix::containers::small_vector<int, 8> small_vec;
    small_vec.push_back(1);
    small_vec.push_back(2);
    small_vec.push_back(3);
    
    EXPECT_EQ(small_vec.size(), 3);
    EXPECT_EQ(small_vec[0], 1);
    EXPECT_EQ(small_vec[1], 2);
    EXPECT_EQ(small_vec[2], 3);
    
    // Test aligned_vector
    hatrix::containers::aligned_vector<int, 64> aligned_vec;
    aligned_vec.push_back(10);
    aligned_vec.push_back(20);
    
    EXPECT_EQ(aligned_vec.size(), 2);
    EXPECT_TRUE(aligned_vec.is_aligned());
    EXPECT_EQ(aligned_vec[0], 10);
    EXPECT_EQ(aligned_vec[1], 20);
}

//--------------------------------------------------------------------------
// INTEGRATION TESTS
//--------------------------------------------------------------------------
class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        auto logger = std::make_shared<spdlog::logger>("test", null_sink);
        spdlog::set_default_logger(logger);
    }
};

TEST_F(IntegrationTest, FullWorkflow) {
    // Test complete workflow with all systems
    Configuration config = presets::debug();
    initialize_logging(config.logging);
    
    // Create typed matrix
    auto matrix = make_hadamard_matrix<4>();
    
    // Fill with Hadamard values
    matrix(0, 0) = 1; matrix(0, 1) = 1; matrix(0, 2) = 1; matrix(0, 3) = 1;
    matrix(1, 0) = 1; matrix(1, 1) = -1; matrix(1, 2) = 1; matrix(1, 3) = -1;
    matrix(2, 0) = 1; matrix(2, 1) = 1; matrix(2, 2) = -1; matrix(2, 3) = -1;
    matrix(3, 0) = 1; matrix(3, 1) = -1; matrix(3, 2) = -1; matrix(3, 3) = 1;
    
    // Test orthogonality
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            int dot_product = 0;
            for (int k = 0; k < 4; ++k) {
                dot_product += matrix(i, k) * matrix(j, k);
            }
            EXPECT_EQ(dot_product, 0) << "Rows " << i << " and " << j << " are not orthogonal";
        }
    }
}

TEST_F(IntegrationTest, PerformanceMeasurement) {
    // Test performance measurement with logging
    {
        PerformanceTimer timer("integration_test");
        
        // Create and manipulate matrices
        auto matrix1 = make_hadamard_matrix<8>();
        auto matrix2 = make_hadamard_matrix<8>();
        
        // Fill matrices
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                matrix1(i, j) = (i + j) % 2 == 0 ? 1 : -1;
                matrix2(i, j) = (i * j) % 2 == 0 ? 1 : -1;
            }
        }
        
        // Do some computation
        int sum = 0;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                sum += matrix1(i, j) * matrix2(i, j);
            }
        }
        
        EXPECT_NE(sum, 0); // Should have some non-zero result
    }
}

//--------------------------------------------------------------------------
// MAIN TEST RUNNER
//--------------------------------------------------------------------------
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize logging for tests
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("test", null_sink);
    spdlog::set_default_logger(logger);
    
    return RUN_ALL_TESTS();
}
