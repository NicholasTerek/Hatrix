// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#pragma once

#include "Hatrix/Export.h"

#include <cstddef>

extern "C" {

typedef struct HatrixMatrixHandle HatrixMatrixHandle;

HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_create(std::size_t rows, std::size_t cols);
HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_create_from_data(
    std::size_t rows,
    std::size_t cols,
    const double* data,
    std::size_t length);
HATRIX_EXPORT void hatrix_matrix_destroy(HatrixMatrixHandle* handle);

HATRIX_EXPORT std::size_t hatrix_matrix_rows(const HatrixMatrixHandle* handle);
HATRIX_EXPORT std::size_t hatrix_matrix_cols(const HatrixMatrixHandle* handle);
HATRIX_EXPORT double hatrix_matrix_get(
    const HatrixMatrixHandle* handle,
    std::size_t row,
    std::size_t col);
HATRIX_EXPORT int hatrix_matrix_set(
    HatrixMatrixHandle* handle,
    std::size_t row,
    std::size_t col,
    double value);
HATRIX_EXPORT int hatrix_matrix_swap_rows(
    HatrixMatrixHandle* handle,
    std::size_t first,
    std::size_t second);
HATRIX_EXPORT int hatrix_matrix_swap_cols(
    HatrixMatrixHandle* handle,
    std::size_t first,
    std::size_t second);
HATRIX_EXPORT int hatrix_matrix_negate_row(HatrixMatrixHandle* handle, std::size_t row);
HATRIX_EXPORT int hatrix_matrix_negate_col(HatrixMatrixHandle* handle, std::size_t col);

HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_add(
    const HatrixMatrixHandle* left,
    const HatrixMatrixHandle* right);
HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_multiply(
    const HatrixMatrixHandle* left,
    const HatrixMatrixHandle* right);
HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_multiply_loop_reordered(
    const HatrixMatrixHandle* left,
    const HatrixMatrixHandle* right);
HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_transpose(const HatrixMatrixHandle* handle);
HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_kronecker(
    const HatrixMatrixHandle* left,
    const HatrixMatrixHandle* right);
HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_normalize(const HatrixMatrixHandle* handle);
HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_sylvester(std::size_t power);
HATRIX_EXPORT int hatrix_matrix_is_hadamard(const HatrixMatrixHandle* handle);
HATRIX_EXPORT int hatrix_matrix_save(const HatrixMatrixHandle* handle, const char* path);
HATRIX_EXPORT HatrixMatrixHandle* hatrix_matrix_load(const char* path);

HATRIX_EXPORT int hatrix_matrix_copy_data(
    const HatrixMatrixHandle* handle,
    double* out_data,
    std::size_t length);

HATRIX_EXPORT const char* hatrix_last_error_message();
HATRIX_EXPORT void hatrix_clear_error();

}
