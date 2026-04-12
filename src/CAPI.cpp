// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#include "Hatrix/CAPI.h"
#include "Hatrix/Matrix.h"

#include <stdexcept>
#include <string>
#include <vector>

struct HatrixMatrixHandle {
    Hatrix::Matrix matrix;
};

namespace {

thread_local std::string g_last_error;

void clear_error() {
    g_last_error.clear();
}

void set_error(const std::exception& error) {
    g_last_error = error.what();
}

void set_unknown_error() {
    g_last_error = "unknown error";
}

}

extern "C" {

HatrixMatrixHandle* hatrix_matrix_create(std::size_t rows, std::size_t cols) {
    clear_error();

    try {
        return new HatrixMatrixHandle{Hatrix::Matrix(rows, cols)};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

HatrixMatrixHandle* hatrix_matrix_create_from_data(
    std::size_t rows,
    std::size_t cols,
    const double* data,
    std::size_t length) {
    clear_error();

    try {
        if (data == nullptr && length != 0) {
            throw std::invalid_argument("matrix data pointer is null");
        }

        std::vector<double> values(data, data + length);
        return new HatrixMatrixHandle{Hatrix::Matrix(rows, cols, values)};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

void hatrix_matrix_destroy(HatrixMatrixHandle* handle) {
    delete handle;
}

std::size_t hatrix_matrix_rows(const HatrixMatrixHandle* handle) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return handle->matrix.rows();
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

std::size_t hatrix_matrix_cols(const HatrixMatrixHandle* handle) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return handle->matrix.cols();
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

double hatrix_matrix_get(const HatrixMatrixHandle* handle, std::size_t row, std::size_t col) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return handle->matrix.get(row, col);
    } catch (const std::exception& error) {
        set_error(error);
        return 0.0;
    } catch (...) {
        set_unknown_error();
        return 0.0;
    }
}

int hatrix_matrix_set(
    HatrixMatrixHandle* handle,
    std::size_t row,
    std::size_t col,
    double value) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        handle->matrix.set(row, col, value);
        return 1;
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

int hatrix_matrix_swap_rows(
    HatrixMatrixHandle* handle,
    std::size_t first,
    std::size_t second) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        handle->matrix.swap_rows(first, second);
        return 1;
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

int hatrix_matrix_swap_cols(
    HatrixMatrixHandle* handle,
    std::size_t first,
    std::size_t second) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        handle->matrix.swap_cols(first, second);
        return 1;
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

int hatrix_matrix_negate_row(HatrixMatrixHandle* handle, std::size_t row) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        handle->matrix.negate_row(row);
        return 1;
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

int hatrix_matrix_negate_col(HatrixMatrixHandle* handle, std::size_t col) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        handle->matrix.negate_col(col);
        return 1;
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

HatrixMatrixHandle* hatrix_matrix_add(
    const HatrixMatrixHandle* left,
    const HatrixMatrixHandle* right) {
    clear_error();

    try {
        if (left == nullptr || right == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return new HatrixMatrixHandle{left->matrix.add(right->matrix)};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

HatrixMatrixHandle* hatrix_matrix_multiply(
    const HatrixMatrixHandle* left,
    const HatrixMatrixHandle* right) {
    clear_error();

    try {
        if (left == nullptr || right == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return new HatrixMatrixHandle{left->matrix.multiply(right->matrix)};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

HatrixMatrixHandle* hatrix_matrix_multiply_loop_reordered(
    const HatrixMatrixHandle* left,
    const HatrixMatrixHandle* right) {
    clear_error();

    try {
        if (left == nullptr || right == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return new HatrixMatrixHandle{
            left->matrix.multiply_loop_reordered(right->matrix)};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

HatrixMatrixHandle* hatrix_matrix_transpose(const HatrixMatrixHandle* handle) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return new HatrixMatrixHandle{handle->matrix.transpose()};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

HatrixMatrixHandle* hatrix_matrix_kronecker(
    const HatrixMatrixHandle* left,
    const HatrixMatrixHandle* right) {
    clear_error();

    try {
        if (left == nullptr || right == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return new HatrixMatrixHandle{left->matrix.kronecker(right->matrix)};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

HatrixMatrixHandle* hatrix_matrix_normalize(const HatrixMatrixHandle* handle) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return new HatrixMatrixHandle{handle->matrix.normalize()};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

HatrixMatrixHandle* hatrix_matrix_sylvester(std::size_t power) {
    clear_error();

    try {
        return new HatrixMatrixHandle{Hatrix::Matrix::sylvester(power)};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

int hatrix_matrix_is_hadamard(const HatrixMatrixHandle* handle) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        return handle->matrix.is_hadamard() ? 1 : 0;
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

int hatrix_matrix_save(const HatrixMatrixHandle* handle, const char* path) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        if (path == nullptr) {
            throw std::invalid_argument("path is null");
        }
        handle->matrix.save(path);
        return 1;
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

HatrixMatrixHandle* hatrix_matrix_load(const char* path) {
    clear_error();

    try {
        if (path == nullptr) {
            throw std::invalid_argument("path is null");
        }
        return new HatrixMatrixHandle{Hatrix::Matrix::load(path)};
    } catch (const std::exception& error) {
        set_error(error);
        return nullptr;
    } catch (...) {
        set_unknown_error();
        return nullptr;
    }
}

int hatrix_matrix_copy_data(
    const HatrixMatrixHandle* handle,
    double* out_data,
    std::size_t length) {
    clear_error();

    try {
        if (handle == nullptr) {
            throw std::invalid_argument("matrix handle is null");
        }
        if (out_data == nullptr && length != 0) {
            throw std::invalid_argument("output buffer is null");
        }
        const auto& values = handle->matrix.values();
        if (length != values.size()) {
            throw std::invalid_argument("output buffer length does not match matrix size");
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            out_data[i] = values[i];
        }
        return 1;
    } catch (const std::exception& error) {
        set_error(error);
        return 0;
    } catch (...) {
        set_unknown_error();
        return 0;
    }
}

const char* hatrix_last_error_message() {
    return g_last_error.c_str();
}

void hatrix_clear_error() {
    g_last_error.clear();
}

}
