// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace Hatrix {

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, const std::vector<double>& values);

    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;

    double get(std::size_t row, std::size_t col) const;
    void set(std::size_t row, std::size_t col, double value);
    void swap_rows(std::size_t first, std::size_t second);
    void swap_cols(std::size_t first, std::size_t second);
    void negate_row(std::size_t row);
    void negate_col(std::size_t col);

    Matrix add(const Matrix& other) const;
    Matrix multiply(const Matrix& other) const;
    Matrix transpose() const;
    Matrix kronecker(const Matrix& other) const;
    Matrix normalize() const;
    bool is_hadamard() const;
    void save(const std::string& path) const;

    static Matrix sylvester(std::size_t power);
    static Matrix load(const std::string& path);

    const std::vector<double>& values() const noexcept;
    std::string repr() const;

private:
    std::size_t index(std::size_t row, std::size_t col) const;

    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> values_;
};

}  // namespace Hatrix
