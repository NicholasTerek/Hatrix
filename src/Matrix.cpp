// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#include "Hatrix/Matrix.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Hatrix {

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), values_(rows * cols, 0.0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, const std::vector<double>& values)
    : rows_(rows), cols_(cols), values_(values) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument("matrix data does not match dimensions");
    }
}

std::size_t Matrix::rows() const noexcept { return rows_; }

std::size_t Matrix::cols() const noexcept { return cols_; }

double Matrix::get(std::size_t row, std::size_t col) const {
    return values_[index(row, col)];
}

void Matrix::set(std::size_t row, std::size_t col, double value) {
    values_[index(row, col)] = value;
}

void Matrix::swap_rows(std::size_t first, std::size_t second) {
    if (first >= rows_ || second >= rows_) {
        throw std::out_of_range("row index out of range");
    }

    for (std::size_t col = 0; col < cols_; ++col) {
        const std::size_t left = first * cols_ + col;
        const std::size_t right = second * cols_ + col;
        const double temp = values_[left];
        values_[left] = values_[right];
        values_[right] = temp;
    }
}

void Matrix::swap_cols(std::size_t first, std::size_t second) {
    if (first >= cols_ || second >= cols_) {
        throw std::out_of_range("column index out of range");
    }

    for (std::size_t row = 0; row < rows_; ++row) {
        const std::size_t left = row * cols_ + first;
        const std::size_t right = row * cols_ + second;
        const double temp = values_[left];
        values_[left] = values_[right];
        values_[right] = temp;
    }
}

void Matrix::negate_row(std::size_t row) {
    if (row >= rows_) {
        throw std::out_of_range("row index out of range");
    }

    for (std::size_t col = 0; col < cols_; ++col) {
        values_[row * cols_ + col] *= -1.0;
    }
}

void Matrix::negate_col(std::size_t col) {
    if (col >= cols_) {
        throw std::out_of_range("column index out of range");
    }

    for (std::size_t row = 0; row < rows_; ++row) {
        values_[row * cols_ + col] *= -1.0;
    }
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("matrix dimensions must match for addition");
    }

    std::vector<double> result(values_.size(), 0.0);
    for (std::size_t i = 0; i < values_.size(); ++i) {
        result[i] = values_[i] + other.values_[i];
    }
    return Matrix(rows_, cols_, result);
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("matrix dimensions are incompatible for multiplication");
    }

    Matrix result(rows_, other.cols_);
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < other.cols_; ++col) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < cols_; ++inner) {
                sum += values_[row * cols_ + inner] *
                       other.values_[inner * other.cols_ + col];
            }
            result.values_[row * other.cols_ + col] = sum;
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            result.values_[col * rows_ + row] = values_[row * cols_ + col];
        }
    }
    return result;
}

Matrix Matrix::kronecker(const Matrix& other) const {
    Matrix result(rows_ * other.rows_, cols_ * other.cols_);

    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            const double scale = values_[row * cols_ + col];
            for (std::size_t other_row = 0; other_row < other.rows_; ++other_row) {
                for (std::size_t other_col = 0; other_col < other.cols_; ++other_col) {
                    const std::size_t result_row = row * other.rows_ + other_row;
                    const std::size_t result_col = col * other.cols_ + other_col;
                    result.values_[result_row * result.cols_ + result_col] =
                        scale * other.values_[other_row * other.cols_ + other_col];
                }
            }
        }
    }

    return result;
}

Matrix Matrix::normalize() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("hadamard normalization requires a square matrix");
    }
    if (rows_ == 0) {
        throw std::invalid_argument("hadamard normalization requires a non-empty matrix");
    }

    Matrix result(*this);

    for (std::size_t col = 0; col < result.cols_; ++col) {
        if (result.values_[col] < 0.0) {
            for (std::size_t row = 0; row < result.rows_; ++row) {
                result.values_[row * result.cols_ + col] *= -1.0;
            }
        }
    }

    for (std::size_t row = 0; row < result.rows_; ++row) {
        if (result.values_[row * result.cols_] < 0.0) {
            for (std::size_t col = 0; col < result.cols_; ++col) {
                result.values_[row * result.cols_ + col] *= -1.0;
            }
        }
    }

    return result;
}

bool Matrix::is_hadamard() const {
    if (rows_ == 0 || rows_ != cols_) {
        return false;
    }

    for (double value : values_) {
        if (value != 1.0 && value != -1.0) {
            return false;
        }
    }

    for (std::size_t left_row = 0; left_row < rows_; ++left_row) {
        for (std::size_t right_row = 0; right_row < rows_; ++right_row) {
            double dot = 0.0;
            for (std::size_t col = 0; col < cols_; ++col) {
                dot += values_[left_row * cols_ + col] * values_[right_row * cols_ + col];
            }

            if (left_row == right_row) {
                if (dot != static_cast<double>(cols_)) {
                    return false;
                }
            } else if (dot != 0.0) {
                return false;
            }
        }
    }

    return true;
}

void Matrix::save(const std::string& path) const {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("could not open file for writing");
    }

    stream << rows_ << ' ' << cols_ << '\n';
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            if (col != 0) {
                stream << ' ';
            }
            stream << values_[row * cols_ + col];
        }
        stream << '\n';
    }
}

Matrix Matrix::sylvester(std::size_t power) {
    Matrix result(1, 1, std::vector<double>{1.0});

    for (std::size_t i = 0; i < power; ++i) {
        Matrix next(result.rows_ * 2, result.cols_ * 2);
        for (std::size_t row = 0; row < result.rows_; ++row) {
            for (std::size_t col = 0; col < result.cols_; ++col) {
                const double value = result.values_[row * result.cols_ + col];
                next.values_[row * next.cols_ + col] = value;
                next.values_[row * next.cols_ + (col + result.cols_)] = value;
                next.values_[(row + result.rows_) * next.cols_ + col] = value;
                next.values_[(row + result.rows_) * next.cols_ + (col + result.cols_)] =
                    -value;
            }
        }
        result = next;
    }

    return result;
}

Matrix Matrix::load(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("could not open file for reading");
    }

    std::size_t rows = 0;
    std::size_t cols = 0;
    stream >> rows >> cols;
    if (!stream) {
        throw std::runtime_error("could not read matrix dimensions");
    }

    std::vector<double> values(rows * cols, 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        stream >> values[i];
        if (!stream) {
            throw std::runtime_error("could not read matrix values");
        }
    }

    return Matrix(rows, cols, values);
}

const std::vector<double>& Matrix::values() const noexcept { return values_; }

std::string Matrix::repr() const {
    std::ostringstream stream;
    stream << "Matrix(" << rows_ << "x" << cols_ << ", [";
    for (std::size_t row = 0; row < rows_; ++row) {
        if (row != 0) {
            stream << ", ";
        }
        stream << "[";
        for (std::size_t col = 0; col < cols_; ++col) {
            if (col != 0) {
                stream << ", ";
            }
            stream << get(row, col);
        }
        stream << "]";
    }
    stream << "])";
    return stream.str();
}

std::size_t Matrix::index(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("matrix index out of range");
    }
    return row * cols_ + col;
}

}  // namespace Hatrix
