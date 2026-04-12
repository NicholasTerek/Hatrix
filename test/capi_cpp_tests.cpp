// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#include "Hatrix/CAPI.h"

#include <cstdlib>
#include <cstring>
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

void clear_error() {
    hatrix_clear_error();
}

std::string last_error() {
    const char* message = hatrix_last_error_message();
    return message == nullptr ? "" : std::string(message);
}

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(int actual, int expected, const std::string& message) {
    if (actual != expected) {
        throw std::runtime_error(message);
    }
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

void require_error_contains(const std::string& needle, const std::string& message) {
    require_true(last_error().find(needle) != std::string::npos, message);
}

struct HandleGuard {
    HatrixMatrixHandle* handle;
    ~HandleGuard() { hatrix_matrix_destroy(handle); }
};

void test_create_and_copy_data_round_trip() {
    const double values[] = {1.0, -1.0, -1.0, 1.0};
    HandleGuard matrix{hatrix_matrix_create_from_data(2, 2, values, 4)};
    require_true(matrix.handle != nullptr, "create_from_data should succeed");

    double copied[4] = {0.0, 0.0, 0.0, 0.0};
    require_equal(hatrix_matrix_copy_data(matrix.handle, copied, 4), 1, "copy_data should succeed");
    require_equal(copied[1], -1.0, "copy_data should preserve values");
}

void test_create_from_data_rejects_null_pointer() {
    clear_error();
    HandleGuard matrix{hatrix_matrix_create_from_data(2, 2, nullptr, 4)};
    require_true(matrix.handle == nullptr, "create_from_data should reject null pointer");
    require_error_contains("matrix data pointer is null", "create_from_data should report null pointer");
}

void test_rows_rejects_null_handle() {
    clear_error();
    require_equal(hatrix_matrix_rows(nullptr), std::size_t{0}, "rows should fail on null handle");
    require_error_contains("matrix handle is null", "rows should report null handle");
}

void test_swap_rows_rejects_out_of_range_index() {
    HandleGuard matrix{hatrix_matrix_create(2, 2)};
    clear_error();
    require_equal(hatrix_matrix_swap_rows(matrix.handle, 0, 5), 0, "swap_rows should fail on bad index");
    require_error_contains("row index out of range", "swap_rows should report bad index");
}

void test_normalize_rejects_non_square_matrix() {
    HandleGuard matrix{hatrix_matrix_create(2, 3)};
    clear_error();
    HandleGuard normalized{hatrix_matrix_normalize(matrix.handle)};
    require_true(normalized.handle == nullptr, "normalize should reject non-square matrices");
    require_error_contains(
        "hadamard normalization requires a square matrix",
        "normalize should report non-square input");
}

void test_is_hadamard_returns_zero_for_non_hadamard_matrix() {
    const double values[] = {1.0, 1.0, 1.0, 1.0};
    HandleGuard matrix{hatrix_matrix_create_from_data(2, 2, values, 4)};
    require_equal(hatrix_matrix_is_hadamard(matrix.handle), 0, "is_hadamard should reject invalid matrices");
}

void test_save_and_load_round_trip() {
    const auto path = (std::filesystem::temp_directory_path() / "hatrix_capi_matrix_io.txt").string();
    const double values[] = {1.0, -1.0, -1.0, 1.0};
    HandleGuard matrix{hatrix_matrix_create_from_data(2, 2, values, 4)};
    require_equal(hatrix_matrix_save(matrix.handle, path.c_str()), 1, "save should succeed");

    HandleGuard loaded{hatrix_matrix_load(path.c_str())};
    std::filesystem::remove(path);

    require_true(loaded.handle != nullptr, "load should succeed");
    require_equal(hatrix_matrix_get(loaded.handle, 1, 1), 1.0, "load should preserve values");
}

void test_load_rejects_missing_file() {
    clear_error();
    HandleGuard loaded{hatrix_matrix_load("/tmp/hatrix_capi_missing_file.txt")};
    require_true(loaded.handle == nullptr, "load should reject missing file");
    require_error_contains("could not open file for reading", "load should report missing file");
}

}  // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"create_and_copy_data_round_trip", test_create_and_copy_data_round_trip},
        {"create_from_data_rejects_null_pointer", test_create_from_data_rejects_null_pointer},
        {"rows_rejects_null_handle", test_rows_rejects_null_handle},
        {"swap_rows_rejects_out_of_range_index", test_swap_rows_rejects_out_of_range_index},
        {"normalize_rejects_non_square_matrix", test_normalize_rejects_non_square_matrix},
        {"is_hadamard_returns_zero_for_non_hadamard_matrix", test_is_hadamard_returns_zero_for_non_hadamard_matrix},
        {"save_and_load_round_trip", test_save_and_load_round_trip},
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
