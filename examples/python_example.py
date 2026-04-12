# Hatrix
# Copyright (c) 2025 Hatrix contributors
# Licensed under the MIT License. See LICENSE for details.

from hatrix import Matrix


def main() -> None:
    left = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    right = Matrix(2, 2, [5.0, 6.0, 7.0, 8.0])

    print("left =", left.to_list())
    print("right =", right.to_list())
    print("left + right =", (left + right).to_list())
    print("left @ right =", (left @ right).to_list())

    hadamard = Matrix.sylvester(2)
    print("sylvester(2) =", hadamard.to_list())
    print("is_hadamard =", hadamard.is_hadamard())

    normalized = Matrix(2, 2, [-1.0, 1.0, 1.0, 1.0]).normalize()
    print("normalized =", normalized.to_list())

    ops = Matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ops.swap_rows(0, 1)
    ops.swap_cols(0, 2)
    ops.negate_row(0)
    ops.negate_col(1)
    ops[1, 2] = 99.0
    print("row/col ops =", ops.to_list())
    print("ops[1, 2] =", ops[1, 2])
    print("ops.shape =", ops.shape)

    output_path = "example_matrix.txt"
    hadamard.save(output_path)
    loaded = Matrix.load(output_path)
    print("loaded from file =", loaded.to_list())


if __name__ == "__main__":
    main()
