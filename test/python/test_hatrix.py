# Hatrix
# Copyright (c) 2025 Hatrix contributors
# Licensed under the MIT License. See LICENSE for details.

import unittest
from pathlib import Path
import tempfile

from hatrix import Matrix


class MatrixConstructionTests(unittest.TestCase):
    def assertMatrixEqual(self, matrix: Matrix, expected: list[list[float]]) -> None:
        self.assertEqual(matrix.to_list(), expected)

    def test_constructs_zero_initialized_matrix(self) -> None:
        matrix = Matrix(2, 3)
        self.assertMatrixEqual(matrix, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def test_addition(self) -> None:
        left = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        right = Matrix(2, 2, [5.0, 6.0, 7.0, 8.0])
        self.assertMatrixEqual(left.add(right), [[6.0, 8.0], [10.0, 12.0]])

    def test_addition_dimension_mismatch_raises(self) -> None:
        left = Matrix(2, 2)
        right = Matrix(2, 3)
        with self.assertRaises(RuntimeError):
            left.add(right)

    def test_invalid_data_length_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            Matrix(2, 2, [1.0, 2.0, 3.0])


class MatrixOperationTests(unittest.TestCase):
    def assertMatrixEqual(self, matrix: Matrix, expected: list[list[float]]) -> None:
        self.assertEqual(matrix.to_list(), expected)

    def test_multiplication(self) -> None:
        left = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        right = Matrix(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        self.assertMatrixEqual(left.multiply(right), [[58.0, 64.0], [139.0, 154.0]])

    def test_set_and_get(self) -> None:
        matrix = Matrix(2, 2)
        matrix.set(1, 0, -7.5)
        self.assertEqual(matrix.get(1, 0), -7.5)
        self.assertMatrixEqual(matrix, [[0.0, 0.0], [-7.5, 0.0]])

    def test_multiplication_dimension_mismatch_raises(self) -> None:
        left = Matrix(2, 2)
        right = Matrix(4, 1)

        with self.assertRaises(RuntimeError):
            left.multiply(right)

    def test_out_of_range_access_raises(self) -> None:
        matrix = Matrix(2, 2)
        with self.assertRaises(RuntimeError):
            matrix.get(10, 0)

    def test_transpose(self) -> None:
        matrix = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertMatrixEqual(matrix.transpose(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_kronecker(self) -> None:
        left = Matrix(2, 2, [1.0, -1.0, 1.0, 1.0])
        right = Matrix(2, 2, [1.0, 1.0, 1.0, -1.0])
        self.assertMatrixEqual(
            left.kronecker(right),
            [
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
            ],
        )

    def test_repr_includes_shape_and_data(self) -> None:
        matrix = Matrix(1, 2, [3.0, 4.0])
        self.assertEqual(repr(matrix), "Matrix(rows=1, cols=2, data=[[3.0, 4.0]])")


class HadamardTests(unittest.TestCase):
    def assertMatrixEqual(self, matrix: Matrix, expected: list[list[float]]) -> None:
        self.assertEqual(matrix.to_list(), expected)

    def test_sylvester_is_hadamard(self) -> None:
        matrix = Matrix.sylvester(2)
        self.assertMatrixEqual(
            matrix,
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
        )
        self.assertTrue(matrix.is_hadamard())

    def test_normalize(self) -> None:
        matrix = Matrix(2, 2, [-1.0, 1.0, 1.0, 1.0])
        self.assertMatrixEqual(matrix.normalize(), [[1.0, 1.0], [1.0, -1.0]])

    def test_normalize_rejects_non_square_matrix(self) -> None:
        with self.assertRaises(RuntimeError):
            Matrix(2, 3).normalize()

    def test_non_hadamard_matrix_returns_false(self) -> None:
        matrix = Matrix(2, 2, [1.0, 1.0, 1.0, 1.0])
        self.assertFalse(matrix.is_hadamard())

    def test_non_pm_one_matrix_returns_false(self) -> None:
        matrix = Matrix(2, 2, [1.0, 2.0, -1.0, 1.0])
        self.assertFalse(matrix.is_hadamard())


class RowAndColumnOperationTests(unittest.TestCase):
    def assertMatrixEqual(self, matrix: Matrix, expected: list[list[float]]) -> None:
        self.assertEqual(matrix.to_list(), expected)

    def test_row_and_column_operations(self) -> None:
        matrix = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        matrix.swap_rows(0, 1)
        matrix.swap_cols(0, 2)
        matrix.negate_row(0)
        matrix.negate_col(1)
        self.assertMatrixEqual(matrix, [[-6.0, 5.0, -4.0], [3.0, -2.0, 1.0]])

    def test_swap_rows_rejects_invalid_index(self) -> None:
        matrix = Matrix(2, 2)
        with self.assertRaises(RuntimeError):
            matrix.swap_rows(0, 5)

    def test_swap_cols_rejects_invalid_index(self) -> None:
        matrix = Matrix(2, 2)
        with self.assertRaises(RuntimeError):
            matrix.swap_cols(0, 5)

    def test_negate_row_rejects_invalid_index(self) -> None:
        matrix = Matrix(2, 2)
        with self.assertRaises(RuntimeError):
            matrix.negate_row(5)

    def test_negate_col_rejects_invalid_index(self) -> None:
        matrix = Matrix(2, 2)
        with self.assertRaises(RuntimeError):
            matrix.negate_col(5)


class PythonErgonomicsTests(unittest.TestCase):
    def assertMatrixEqual(self, matrix: Matrix, expected: list[list[float]]) -> None:
        self.assertEqual(matrix.to_list(), expected)

    def test_python_ergonomics(self) -> None:
        left = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
        right = Matrix(2, 2, [5.0, 6.0, 7.0, 8.0])

        left[0, 1] = 10.0

        self.assertEqual(left[0, 1], 10.0)
        self.assertEqual(left.shape, (2, 2))
        self.assertEqual(len(left), 2)
        self.assertMatrixEqual(left + right, [[6.0, 16.0], [10.0, 12.0]])
        self.assertMatrixEqual(left @ right, [[75.0, 86.0], [43.0, 50.0]])


class MatrixIoTests(unittest.TestCase):
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "matrix_python_io.txt"
            matrix = Matrix.sylvester(2)

            matrix.save(path)
            loaded = Matrix.load(path)

            self.assertEqual(loaded.to_list(), matrix.to_list())

    def test_load_rejects_missing_file(self) -> None:
        with self.assertRaises(RuntimeError):
            Matrix.load("/tmp/hatrix_python_missing_file.txt")


if __name__ == "__main__":
    unittest.main()
