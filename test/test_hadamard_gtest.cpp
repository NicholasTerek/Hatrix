//======================================================================
// test_hadamard_gtest.cpp
//----------------------------------------------------------------------
// Google Test unit tests for the Hadamard matrix library.
// Comprehensive test suite covering all functionality.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================

#include <gtest/gtest.h>
#include "../Hatrix/hadamard_matrix.hpp"
#include <vector>
#include <cmath>
#include <random>

class HadamardTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up common test data
        rng_.seed(42);
    }
    
    void TearDown() override {
        // Clean up after tests
    }
    
    std::mt19937 rng_;
    
    // Helper function to check if two vectors are approximately equal
    bool vectors_approx_equal(const std::vector<double>& a, const std::vector<double>& b, double tolerance = 1e-10) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tolerance) return false;
        }
        return true;
    }
    
    // Helper function to check if two matrices are equal
    bool matrices_equal(const hadamard::matrix_t& a, const hadamard::matrix_t& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].size() != b[i].size()) return false;
            for (size_t j = 0; j < a[i].size(); ++j) {
                if (a[i][j] != b[i][j]) return false;
            }
        }
        return true;
    }
};

//--------------------------------------------------------------------------
// ORDER VALIDATION TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, ValidateOrder_ValidOrders) {
    // Test valid power-of-two orders
    std::vector<int> valid_orders = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    for (int order : valid_orders) {
        EXPECT_NO_THROW(hadamard::validate_order(order));
    }
}

TEST_F(HadamardTest, ValidateOrder_InvalidOrders) {
    // Test invalid orders
    std::vector<int> invalid_orders = {0, -1, 3, 5, 6, 7, 9, 10, 15, 33, 100};
    
    for (int order : invalid_orders) {
        EXPECT_THROW(hadamard::validate_order(order), std::invalid_argument);
    }
}

//--------------------------------------------------------------------------
// MATRIX GENERATION TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, GenerateRecursive_BasicCases) {
    // Test H(1)
    auto H1 = hadamard::generate_recursive(1);
    EXPECT_EQ(1, static_cast<int>(H1.size()));
    EXPECT_EQ(1, H1[0][0]);
    
    // Test H(2)
    auto H2 = hadamard::generate_recursive(2);
    EXPECT_EQ(2, static_cast<int>(H2.size()));
    EXPECT_EQ(1, H2[0][0]);
    EXPECT_EQ(1, H2[0][1]);
    EXPECT_EQ(1, H2[1][0]);
    EXPECT_EQ(-1, H2[1][1]);
    
    // Test H(4)
    auto H4 = hadamard::generate_recursive(4);
    EXPECT_EQ(4, static_cast<int>(H4.size()));
    
    // Verify Sylvester construction pattern
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_EQ(H2[i][j], H4[i][j]) << "Top-left quadrant should match H(2)";
            EXPECT_EQ(H2[i][j], H4[i][j+2]) << "Top-right quadrant should match H(2)";
            EXPECT_EQ(H2[i][j], H4[i+2][j]) << "Bottom-left quadrant should match H(2)";
            EXPECT_EQ(-H2[i][j], H4[i+2][j+2]) << "Bottom-right quadrant should be -H(2)";
        }
    }
}

TEST_F(HadamardTest, GenerateIterative_MatchesRecursive) {
    // Test that iterative matches recursive for various sizes
    std::vector<int> sizes = {1, 2, 4, 8, 16, 32, 64};
    
    for (int size : sizes) {
        auto H_rec = hadamard::generate_recursive(size);
        auto H_iter = hadamard::generate_iterative(size);
        
        EXPECT_EQ(H_rec.size(), H_iter.size()) << "Size mismatch for H(" << size << ")";
        EXPECT_TRUE(matrices_equal(H_rec, H_iter)) << "Recursive and iterative should match for H(" << size << ")";
    }
}

TEST_F(HadamardTest, GenerateWalsh_Orderings) {
    auto H4 = hadamard::generate_recursive(4);
    auto W4_natural = hadamard::generate_walsh(4, hadamard::ordering_t::NATURAL);
    auto W4_sequency = hadamard::generate_walsh(4, hadamard::ordering_t::SEQUENCY);
    auto W4_dyadic = hadamard::generate_walsh(4, hadamard::ordering_t::DYADIC);
    
    // Natural ordering should match recursive
    EXPECT_TRUE(matrices_equal(H4, W4_natural)) << "Natural ordering should match recursive";
    
    // All orderings should produce valid Hadamard matrices
    EXPECT_TRUE(hadamard::is_hadamard(W4_sequency)) << "Sequency ordering should produce Hadamard matrix";
    EXPECT_TRUE(hadamard::is_hadamard(W4_dyadic)) << "Dyadic ordering should produce Hadamard matrix";
}

//--------------------------------------------------------------------------
// MATRIX OPERATIONS TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, Transpose) {
    auto H4 = hadamard::generate_recursive(4);
    auto T4 = hadamard::transpose(H4);
    
    EXPECT_EQ(H4.size(), T4.size()) << "Transpose should preserve size";
    
    for (size_t i = 0; i < H4.size(); ++i) {
        for (size_t j = 0; j < H4[i].size(); ++j) {
            EXPECT_EQ(H4[i][j], T4[j][i]) << "Transpose should swap indices at (" << i << "," << j << ")";
        }
    }
    
    // Test with empty matrix
    hadamard::matrix_t empty;
    auto T_empty = hadamard::transpose(empty);
    EXPECT_EQ(0, static_cast<int>(T_empty.size())) << "Transpose of empty should be empty";
}

TEST_F(HadamardTest, MatrixVectorMultiply) {
    auto H2 = hadamard::generate_recursive(2);
    hadamard::vector_t v = {1, 2};
    
    auto result = hadamard::multiply(H2, v);
    EXPECT_EQ(2, static_cast<int>(result.size())) << "Result should have correct size";
    
    // H(2) * [1, 2] = [1+2, 1-2] = [3, -1]
    EXPECT_EQ(3, result[0]) << "Matrix-vector multiply result[0]";
    EXPECT_EQ(-1, result[1]) << "Matrix-vector multiply result[1]";
    
    // Test dimension mismatch
    hadamard::vector_t wrong_size = {1, 2, 3};
    EXPECT_THROW(hadamard::multiply(H2, wrong_size), std::invalid_argument);
}

TEST_F(HadamardTest, MatrixMatrixMultiply) {
    auto H2 = hadamard::generate_recursive(2);
    auto H2T = hadamard::transpose(H2);
    auto H2H2T = hadamard::multiply(H2, H2T);
    
    // For Hadamard matrix: H * H^T = n * I
    EXPECT_EQ(2, H2H2T[0][0]) << "H*H^T diagonal should be n";
    EXPECT_EQ(2, H2H2T[1][1]) << "H*H^T diagonal should be n";
    EXPECT_EQ(0, H2H2T[0][1]) << "H*H^T off-diagonal should be 0";
    EXPECT_EQ(0, H2H2T[1][0]) << "H*H^T off-diagonal should be 0";
    
    // Test dimension mismatch
    hadamard::matrix_t wrong_size = {{1, -1}, {1, -1}, {1, -1}};
    EXPECT_THROW(hadamard::multiply(H2, wrong_size), std::invalid_argument);
}

//--------------------------------------------------------------------------
// VALIDATION TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, IsOrthogonal) {
    // Test valid Hadamard matrices
    std::vector<int> sizes = {2, 4, 8, 16, 32};
    
    for (int size : sizes) {
        auto H = hadamard::generate_recursive(size);
        EXPECT_TRUE(hadamard::is_orthogonal(H)) << "H(" << size << ") should be orthogonal";
    }
    
    // Test invalid matrix
    hadamard::matrix_t invalid = {{1, 2}, {3, 4}};
    EXPECT_FALSE(hadamard::is_orthogonal(invalid)) << "Invalid matrix should not be orthogonal";
}

TEST_F(HadamardTest, IsHadamard) {
    // Test valid Hadamard matrices
    std::vector<int> sizes = {2, 4, 8, 16, 32};
    
    for (int size : sizes) {
        auto H = hadamard::generate_recursive(size);
        EXPECT_TRUE(hadamard::is_hadamard(H)) << "Generated H(" << size << ") should be Hadamard";
    }
    
    // Test invalid matrices
    hadamard::matrix_t not_square = {{1, -1}, {1}};
    EXPECT_FALSE(hadamard::is_hadamard(not_square)) << "Non-square matrix should not be Hadamard";
    
    hadamard::matrix_t not_pm1 = {{1, 2}, {-1, 1}};
    EXPECT_FALSE(hadamard::is_hadamard(not_pm1)) << "Non-±1 matrix should not be Hadamard";
    
    hadamard::matrix_t not_orthogonal = {{1, 1}, {1, 1}};
    EXPECT_FALSE(hadamard::is_hadamard(not_orthogonal)) << "Non-orthogonal matrix should not be Hadamard";
}

TEST_F(HadamardTest, ValidateMatrix) {
    // Test valid matrix
    auto H4 = hadamard::generate_recursive(4);
    auto issues = hadamard::validate_matrix(H4);
    EXPECT_EQ(0, static_cast<int>(issues.size())) << "Valid Hadamard matrix should have no issues";
    
    // Test invalid matrices
    hadamard::matrix_t empty;
    auto empty_issues = hadamard::validate_matrix(empty);
    EXPECT_EQ(1, static_cast<int>(empty_issues.size())) << "Empty matrix should have one issue";
    EXPECT_TRUE(empty_issues[0].find("empty") != std::string::npos) << "Should report empty matrix";
    
    hadamard::matrix_t not_power_of_2 = {{1, -1}, {1, -1}, {1, -1}};
    auto pow2_issues = hadamard::validate_matrix(not_power_of_2);
    EXPECT_GT(pow2_issues.size(), 0) << "Non-power-of-2 matrix should have issues";
}

//--------------------------------------------------------------------------
// TRANSFORM TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, FWHT_Basic) {
    hadamard::dvector_t data = {1.0, 2.0, 3.0, 4.0};
    auto transformed = hadamard::fwht(data);
    
    EXPECT_EQ(data.size(), transformed.size()) << "FWHT should preserve size";
    
    // Test round-trip property: FWHT(FWHT(x)) = x
    auto round_trip = hadamard::fwht(transformed);
    EXPECT_TRUE(vectors_approx_equal(data, round_trip)) << "FWHT round-trip should recover original";
}

TEST_F(HadamardTest, FWHT_VariousSizes) {
    std::vector<int> sizes = {2, 4, 8, 16, 32, 64};
    
    for (int size : sizes) {
        hadamard::dvector_t test_data(size);
        std::iota(test_data.begin(), test_data.end(), 1.0);
        
        auto fwht_result = hadamard::fwht(test_data);
        auto round_trip_result = hadamard::fwht(fwht_result);
        
        EXPECT_TRUE(vectors_approx_equal(test_data, round_trip_result)) 
            << "FWHT round-trip should work for size " << size;
    }
}

TEST_F(HadamardTest, IFWHT) {
    hadamard::dvector_t data = {1.0, 2.0, 3.0, 4.0};
    auto transformed = hadamard::fwht(data);
    auto inverted = hadamard::ifwht(transformed);
    
    EXPECT_TRUE(vectors_approx_equal(data, inverted)) << "IFWHT should recover original data";
}

TEST_F(HadamardTest, FWHT_InvalidSize) {
    hadamard::dvector_t invalid_data = {1.0, 2.0, 3.0};  // Not power of 2
    EXPECT_THROW(hadamard::fwht(invalid_data), std::invalid_argument) << "FWHT should reject non-power-of-2 size";
}

//--------------------------------------------------------------------------
// SERIALIZATION TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, SerializeDeserialize) {
    auto H4 = hadamard::generate_recursive(4);
    
    // Test different formats
    std::vector<hadamard::format_t> formats = {
        hadamard::format_t::COMPACT,
        hadamard::format_t::VERBOSE,
        hadamard::format_t::CSV,
        hadamard::format_t::BINARY
    };
    
    for (auto fmt : formats) {
        std::string serialized = hadamard::serialize(H4, fmt);
        auto deserialized = hadamard::deserialize(serialized, fmt);
        
        EXPECT_EQ(H4.size(), deserialized.size()) << "Deserialized should preserve size";
        EXPECT_TRUE(matrices_equal(H4, deserialized)) << "Deserialized should match original";
    }
}

TEST_F(HadamardTest, FileIO) {
    auto H4 = hadamard::generate_recursive(4);
    const std::string filename = "test_matrix.txt";
    
    try {
        // Save to file
        hadamard::save_to_file(H4, filename, hadamard::format_t::COMPACT);
        
        // Load from file
        auto loaded = hadamard::load_from_file(filename, hadamard::format_t::COMPACT);
        EXPECT_TRUE(matrices_equal(H4, loaded)) << "Loaded matrix should match original";
        
        // Clean up
        std::remove(filename.c_str());
        
    } catch (const std::exception& e) {
        FAIL() << "File I/O should not throw: " << e.what();
    }
}

//--------------------------------------------------------------------------
// PROPERTIES ANALYSIS TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, AnalyzeProperties) {
    auto H4 = hadamard::generate_recursive(4);
    auto props = hadamard::analyze_properties(H4);
    
    EXPECT_DOUBLE_EQ(4.0, props["size"]) << "Size property should be 4";
    EXPECT_DOUBLE_EQ(1.0, props["is_hadamard"]) << "Should be marked as Hadamard";
    EXPECT_DOUBLE_EQ(1.0, props["is_orthogonal"]) << "Should be marked as orthogonal";
    EXPECT_DOUBLE_EQ(16.0, props["expected_det_magnitude"]) << "Expected determinant magnitude should be 16";
    EXPECT_DOUBLE_EQ(2.0, props["expected_condition_number"]) << "Expected condition number should be sqrt(4) = 2";
}

//--------------------------------------------------------------------------
// UTILITY FUNCTION TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, GrayCodeConversion) {
    // Test basic Gray code conversion
    EXPECT_EQ(0, hadamard::binary_to_gray(0)) << "Gray code of 0 should be 0";
    EXPECT_EQ(1, hadamard::binary_to_gray(1)) << "Gray code of 1 should be 1";
    EXPECT_EQ(3, hadamard::binary_to_gray(2)) << "Gray code of 2 should be 3";
    EXPECT_EQ(2, hadamard::binary_to_gray(3)) << "Gray code of 3 should be 2";
    
    // Test round-trip conversion
    for (int i = 0; i < 16; ++i) {
        int gray = hadamard::binary_to_gray(i);
        int back = hadamard::gray_to_binary(gray);
        EXPECT_EQ(i, back) << "Gray code round-trip should work for " << i;
    }
}

TEST_F(HadamardTest, Benchmark) {
    double time_us = hadamard::benchmark([]() {
        auto H = hadamard::generate_recursive(16);
        volatile auto _ = hadamard::is_hadamard(H);  // Prevent optimization
    }, 10);
    
    EXPECT_GE(time_us, 0.0) << "Benchmark should return non-negative time";
    EXPECT_LT(time_us, 1000000.0) << "Benchmark should complete in reasonable time";
}

//--------------------------------------------------------------------------
// STRESS TESTS
//--------------------------------------------------------------------------
TEST_F(HadamardTest, LargeMatrixGeneration) {
    // Test generation of larger matrices
    std::vector<int> sizes = {64, 128, 256};
    
    for (int size : sizes) {
        EXPECT_NO_THROW({
            auto H = hadamard::generate_iterative(size);  // Use iterative for large sizes
            EXPECT_TRUE(hadamard::is_hadamard(H)) << "Large H(" << size << ") should be valid Hadamard";
        });
    }
}

TEST_F(HadamardTest, LargeTransform) {
    // Test large transforms
    std::vector<int> sizes = {64, 128, 256, 512, 1024};
    
    for (int size : sizes) {
        hadamard::dvector_t data(size);
        std::iota(data.begin(), data.end(), 1.0);
        
        EXPECT_NO_THROW({
            auto transformed = hadamard::fwht(data);
            auto round_trip = hadamard::fwht(transformed);
            EXPECT_TRUE(vectors_approx_equal(data, round_trip)) << "Large transform should work for size " << size;
        });
    }
}

//--------------------------------------------------------------------------
// PARAMETERIZED TESTS
//--------------------------------------------------------------------------
class HadamardParameterizedTest : public ::testing::TestWithParam<int> {};

TEST_P(HadamardParameterizedTest, ValidHadamardGeneration) {
    int size = GetParam();
    
    auto H = hadamard::generate_recursive(size);
    EXPECT_TRUE(hadamard::is_hadamard(H)) << "H(" << size << ") should be valid Hadamard matrix";
    EXPECT_TRUE(hadamard::is_orthogonal(H)) << "H(" << size << ") should be orthogonal";
}

INSTANTIATE_TEST_SUITE_P(
    ValidSizes,
    HadamardParameterizedTest,
    ::testing::Values(2, 4, 8, 16, 32, 64)
);

//--------------------------------------------------------------------------
// MAIN FUNCTION
//--------------------------------------------------------------------------
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
