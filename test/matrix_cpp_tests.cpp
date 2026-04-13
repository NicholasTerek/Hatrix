// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#include "Hatrix/Matrix.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TestCase {
    const char* name;
    std::function<void()> run;
};

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_false(bool condition, const std::string& message) {
    require_true(!condition, message);
}

void require_equal(std::size_t actual, std::size_t expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(message);
    }
}

void require_equal(double actual, double expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(message);
    }
}

template <typename ExceptionType>
void require_throws(const std::function<void()>& func, const std::string& message) {
    bool threw = false;
    try {
        func();
    } catch (const ExceptionType&) {
        threw = true;
    }
    require_true(threw, message);
}

void require_matrix_equals(
    const Hatrix::Matrix& matrix,
    std::size_t rows,
    std::size_t cols,
    const std::vector<double>& values,
    const std::string& message) {
    require_equal(matrix.rows(), rows, message + ": wrong row count");
    require_equal(matrix.cols(), cols, message + ": wrong column count");
    require_equal(matrix.values().size(), values.size(), message + ": wrong flattened size");

    for (std::size_t i = 0; i < values.size(); ++i) {
        require_equal(matrix.values()[i], values[i], message + ": wrong value");
    }
}

void test_constructs_expected_shape() {
    Hatrix::Matrix matrix(2, 3);
    require_equal(matrix.rows(), 2, "constructor should set row count");
}

void test_constructs_expected_width() {
    Hatrix::Matrix matrix(2, 3);
    require_equal(matrix.cols(), 3, "constructor should set column count");
}

void test_constructs_zero_initialized_values() {
    Hatrix::Matrix matrix(2, 3);
    require_equal(matrix.get(1, 2), 0.0, "constructor should zero-initialize values");
}

void test_constructor_rejects_bad_data_length() {
    require_throws<std::invalid_argument>(
        []() { Hatrix::Matrix(2, 2, std::vector<double>{1.0, 2.0, 3.0}); },
        "constructor should reject mismatched data length");
}

void test_set_updates_value() {
    Hatrix::Matrix matrix(2, 2);
    matrix.set(1, 0, -3.25);
    require_equal(matrix.get(1, 0), -3.25, "set should update the addressed value");
}

void test_get_rejects_out_of_range_index() {
    Hatrix::Matrix matrix(2, 2);
    require_throws<std::out_of_range>(
        [&]() { (void)matrix.get(10, 0); },
        "get should reject out-of-range access");
}

void test_add_returns_expected_values() {
    Hatrix::Matrix left(2, 2, std::vector<double>{1.0, 2.0, 3.0, 4.0});
    Hatrix::Matrix right(2, 2, std::vector<double>{5.0, 6.0, 7.0, 8.0});
    const auto result = left.add(right);
    require_matrix_equals(
        result,
        2,
        2,
        std::vector<double>{6.0, 8.0, 10.0, 12.0},
        "add should sum corresponding entries");
}

void test_add_rejects_mismatched_dimensions() {
    Hatrix::Matrix left(2, 2);
    Hatrix::Matrix right(2, 3);
    require_throws<std::invalid_argument>(
        [&]() { (void)left.add(right); },
        "add should reject mismatched dimensions");
}

void test_multiply_returns_expected_values() {
    Hatrix::Matrix left(2, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Hatrix::Matrix right(3, 2, std::vector<double>{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    const auto result = left.multiply(right);
    require_matrix_equals(
        result,
        2,
        2,
        std::vector<double>{58.0, 64.0, 139.0, 154.0},
        "multiply should produce standard matrix multiplication");
}

void test_multiply_loop_reordered_returns_expected_values() {
    Hatrix::Matrix left(2, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Hatrix::Matrix right(3, 2, std::vector<double>{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    const auto result = left.multiply_loop_reordered(right);
    require_matrix_equals(
        result,
        2,
        2,
        std::vector<double>{58.0, 64.0, 139.0, 154.0},
        "multiply_loop_reordered should produce standard matrix multiplication");
}

void test_multiply_loop_reordered_matches_baseline() {
    Hatrix::Matrix left(3, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    Hatrix::Matrix right(3, 3, std::vector<double>{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    const auto baseline = left.multiply(right);
    const auto reordered = left.multiply_loop_reordered(right);
    require_matrix_equals(
        reordered,
        baseline.rows(),
        baseline.cols(),
        baseline.values(),
        "multiply_loop_reordered should match baseline multiply");
}

void test_multiply_inner_tiled_returns_expected_values() {
    Hatrix::Matrix left(2, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Hatrix::Matrix right(3, 2, std::vector<double>{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    const auto result = left.multiply_inner_tiled(right, 2);
    require_matrix_equals(
        result,
        2,
        2,
        std::vector<double>{58.0, 64.0, 139.0, 154.0},
        "multiply_inner_tiled should produce standard matrix multiplication");
}

void test_multiply_inner_tiled_matches_baseline() {
    Hatrix::Matrix left(3, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    Hatrix::Matrix right(3, 3, std::vector<double>{9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    const auto baseline = left.multiply(right);
    const auto tiled = left.multiply_inner_tiled(right, 2);
    require_matrix_equals(
        tiled,
        baseline.rows(),
        baseline.cols(),
        baseline.values(),
        "multiply_inner_tiled should match baseline multiply");
}

void test_multiply_inner_tiled_rejects_zero_tile_size() {
    Hatrix::Matrix left(2, 2);
    Hatrix::Matrix right(2, 2);
    require_throws<std::invalid_argument>(
        [&]() { (void)left.multiply_inner_tiled(right, 0); },
        "multiply_inner_tiled should reject zero tile size");
}

void test_multiply_rejects_mismatched_dimensions() {
    Hatrix::Matrix left(2, 2);
    Hatrix::Matrix right(3, 2);
    require_throws<std::invalid_argument>(
        [&]() { (void)left.multiply(right); },
        "multiply should reject mismatched dimensions");
}

void test_transpose_swaps_dimensions_and_values() {
    Hatrix::Matrix matrix(2, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const auto transposed = matrix.transpose();
    require_matrix_equals(
        transposed,
        3,
        2,
        std::vector<double>{1.0, 4.0, 2.0, 5.0, 3.0, 6.0},
        "transpose should swap axes");
}

void test_kronecker_expands_shape_and_values() {
    Hatrix::Matrix left(2, 2, std::vector<double>{1.0, -1.0, 1.0, 1.0});
    Hatrix::Matrix right(2, 2, std::vector<double>{1.0, 1.0, 1.0, -1.0});
    const auto result = left.kronecker(right);
    require_matrix_equals(
        result,
        4,
        4,
        std::vector<double>{
            1.0, 1.0, -1.0, -1.0,
            1.0, -1.0, -1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, -1.0, 1.0, -1.0},
        "kronecker should expand block-wise");
}

void test_sylvester_power_zero_returns_one_by_one_matrix() {
    const auto matrix = Hatrix::Matrix::sylvester(0);
    require_matrix_equals(matrix, 1, 1, std::vector<double>{1.0}, "sylvester(0) should be [1]");
}

void test_sylvester_power_two_is_hadamard() {
    const auto matrix = Hatrix::Matrix::sylvester(2);
    require_true(matrix.is_hadamard(), "sylvester matrix should satisfy hadamard conditions");
}

void test_normalize_makes_first_row_positive() {
    Hatrix::Matrix matrix(2, 2, std::vector<double>{-1.0, 1.0, 1.0, 1.0});
    const auto normalized = matrix.normalize();
    require_equal(normalized.get(0, 1), 1.0, "normalize should make first row positive");
}

void test_normalize_makes_first_column_positive() {
    Hatrix::Matrix matrix(2, 2, std::vector<double>{-1.0, 1.0, 1.0, 1.0});
    const auto normalized = matrix.normalize();
    require_equal(normalized.get(1, 0), 1.0, "normalize should make first column positive");
}

void test_normalize_rejects_non_square_matrix() {
    Hatrix::Matrix matrix(2, 3);
    require_throws<std::invalid_argument>(
        [&]() { (void)matrix.normalize(); },
        "normalize should reject non-square matrices");
}

void test_is_hadamard_rejects_non_square_matrix() {
    Hatrix::Matrix matrix(2, 3, std::vector<double>{1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
    require_false(matrix.is_hadamard(), "hadamard check should reject non-square matrices");
}

void test_is_hadamard_rejects_non_pm_one_entries() {
    Hatrix::Matrix matrix(2, 2, std::vector<double>{1.0, 2.0, -1.0, 1.0});
    require_false(matrix.is_hadamard(), "hadamard check should reject non +/-1 entries");
}

void test_is_hadamard_rejects_non_orthogonal_matrix() {
    Hatrix::Matrix matrix(2, 2, std::vector<double>{1.0, 1.0, 1.0, 1.0});
    require_false(matrix.is_hadamard(), "hadamard check should reject non-orthogonal rows");
}

void test_swap_rows_updates_matrix_layout() {
    Hatrix::Matrix matrix(2, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    matrix.swap_rows(0, 1);
    require_matrix_equals(
        matrix,
        2,
        3,
        std::vector<double>{4.0, 5.0, 6.0, 1.0, 2.0, 3.0},
        "swap_rows should exchange complete rows");
}

void test_swap_rows_rejects_out_of_range_index() {
    Hatrix::Matrix matrix(2, 2);
    require_throws<std::out_of_range>(
        [&]() { matrix.swap_rows(0, 3); },
        "swap_rows should reject out-of-range indices");
}

void test_swap_cols_updates_matrix_layout() {
    Hatrix::Matrix matrix(2, 3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    matrix.swap_cols(0, 2);
    require_matrix_equals(
        matrix,
        2,
        3,
        std::vector<double>{3.0, 2.0, 1.0, 6.0, 5.0, 4.0},
        "swap_cols should exchange complete columns");
}

void test_negate_row_updates_signs() {
    Hatrix::Matrix matrix(2, 2, std::vector<double>{1.0, -2.0, 3.0, -4.0});
    matrix.negate_row(1);
    require_matrix_equals(
        matrix,
        2,
        2,
        std::vector<double>{1.0, -2.0, -3.0, 4.0},
        "negate_row should negate all values in the target row");
}

void test_negate_col_updates_signs() {
    Hatrix::Matrix matrix(2, 2, std::vector<double>{1.0, -2.0, 3.0, -4.0});
    matrix.negate_col(1);
    require_matrix_equals(
        matrix,
        2,
        2,
        std::vector<double>{1.0, 2.0, 3.0, 4.0},
        "negate_col should negate all values in the target column");
}

void test_save_and_load_round_trip_values() {
    const auto path = (std::filesystem::temp_directory_path() / "hatrix_test_matrix_io.txt").string();
    Hatrix::Matrix written(2, 2, std::vector<double>{1.0, -1.0, -1.0, 1.0});
    written.save(path);
    const auto loaded = Hatrix::Matrix::load(path);
    std::filesystem::remove(path);

    require_matrix_equals(
        loaded,
        2,
        2,
        std::vector<double>{1.0, -1.0, -1.0, 1.0},
        "save/load should preserve matrix data");
}

void test_load_rejects_missing_file() {
    require_throws<std::runtime_error>(
        []() { (void)Hatrix::Matrix::load("/tmp/hatrix_file_that_should_not_exist.txt"); },
        "load should reject missing files");
}

}  // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"constructs_expected_shape", test_constructs_expected_shape},
        {"constructs_expected_width", test_constructs_expected_width},
        {"constructs_zero_initialized_values", test_constructs_zero_initialized_values},
        {"constructor_rejects_bad_data_length", test_constructor_rejects_bad_data_length},
        {"set_updates_value", test_set_updates_value},
        {"get_rejects_out_of_range_index", test_get_rejects_out_of_range_index},
        {"add_returns_expected_values", test_add_returns_expected_values},
        {"add_rejects_mismatched_dimensions", test_add_rejects_mismatched_dimensions},
        {"multiply_returns_expected_values", test_multiply_returns_expected_values},
        {"multiply_loop_reordered_returns_expected_values", test_multiply_loop_reordered_returns_expected_values},
        {"multiply_loop_reordered_matches_baseline", test_multiply_loop_reordered_matches_baseline},
        {"multiply_inner_tiled_returns_expected_values", test_multiply_inner_tiled_returns_expected_values},
        {"multiply_inner_tiled_matches_baseline", test_multiply_inner_tiled_matches_baseline},
        {"multiply_inner_tiled_rejects_zero_tile_size", test_multiply_inner_tiled_rejects_zero_tile_size},
        {"multiply_rejects_mismatched_dimensions", test_multiply_rejects_mismatched_dimensions},
        {"transpose_swaps_dimensions_and_values", test_transpose_swaps_dimensions_and_values},
        {"kronecker_expands_shape_and_values", test_kronecker_expands_shape_and_values},
        {"sylvester_power_zero_returns_one_by_one_matrix", test_sylvester_power_zero_returns_one_by_one_matrix},
        {"sylvester_power_two_is_hadamard", test_sylvester_power_two_is_hadamard},
        {"normalize_makes_first_row_positive", test_normalize_makes_first_row_positive},
        {"normalize_makes_first_column_positive", test_normalize_makes_first_column_positive},
        {"normalize_rejects_non_square_matrix", test_normalize_rejects_non_square_matrix},
        {"is_hadamard_rejects_non_square_matrix", test_is_hadamard_rejects_non_square_matrix},
        {"is_hadamard_rejects_non_pm_one_entries", test_is_hadamard_rejects_non_pm_one_entries},
        {"is_hadamard_rejects_non_orthogonal_matrix", test_is_hadamard_rejects_non_orthogonal_matrix},
        {"swap_rows_updates_matrix_layout", test_swap_rows_updates_matrix_layout},
        {"swap_rows_rejects_out_of_range_index", test_swap_rows_rejects_out_of_range_index},
        {"swap_cols_updates_matrix_layout", test_swap_cols_updates_matrix_layout},
        {"negate_row_updates_signs", test_negate_row_updates_signs},
        {"negate_col_updates_signs", test_negate_col_updates_signs},
        {"save_and_load_round_trip_values", test_save_and_load_round_trip_values},
        {"load_rejects_missing_file", test_load_rejects_missing_file},
    };

    for (const auto& test : tests) {
        try {
            test.run();
        } catch (const std::exception& error) {
            std::cerr << test.name << ": " << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
