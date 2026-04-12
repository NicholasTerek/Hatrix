# Hatrix
# Copyright (c) 2025 Hatrix contributors
# Licensed under the MIT License. See LICENSE for details.

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _library_filename() -> str:
    if os.name == "nt":
        return "hatrix.dll"
    if sys.platform == "darwin":
        return "libhatrix.dylib"
    if os.name == "posix":
        return "libhatrix.so"
    raise RuntimeError("unsupported platform")


def _load_library() -> ctypes.CDLL:
    package_dir = Path(__file__).resolve().parent
    env_path = Path(__file__).resolve().parent.parent.parent / "build"
    candidates = [
        package_dir / _library_filename(),
        env_path / _library_filename(),
        env_path / "libhatrix.so",
        env_path / "Debug" / _library_filename(),
        env_path / "Release" / _library_filename(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return ctypes.CDLL(str(candidate))

    raise ImportError(
        "Could not find the hatrix shared library. Build the project with CMake first."
    )


_LIB = _load_library()

_LIB.hatrix_matrix_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_LIB.hatrix_matrix_create.restype = ctypes.c_void_p

_LIB.hatrix_matrix_create_from_data.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
_LIB.hatrix_matrix_create_from_data.restype = ctypes.c_void_p

_LIB.hatrix_matrix_destroy.argtypes = [ctypes.c_void_p]
_LIB.hatrix_matrix_destroy.restype = None

_LIB.hatrix_matrix_rows.argtypes = [ctypes.c_void_p]
_LIB.hatrix_matrix_rows.restype = ctypes.c_size_t

_LIB.hatrix_matrix_cols.argtypes = [ctypes.c_void_p]
_LIB.hatrix_matrix_cols.restype = ctypes.c_size_t

_LIB.hatrix_matrix_get.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
_LIB.hatrix_matrix_get.restype = ctypes.c_double

_LIB.hatrix_matrix_set.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_double,
]
_LIB.hatrix_matrix_set.restype = ctypes.c_int

_LIB.hatrix_matrix_swap_rows.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
_LIB.hatrix_matrix_swap_rows.restype = ctypes.c_int

_LIB.hatrix_matrix_swap_cols.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
_LIB.hatrix_matrix_swap_cols.restype = ctypes.c_int

_LIB.hatrix_matrix_negate_row.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_LIB.hatrix_matrix_negate_row.restype = ctypes.c_int

_LIB.hatrix_matrix_negate_col.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_LIB.hatrix_matrix_negate_col.restype = ctypes.c_int

_LIB.hatrix_matrix_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_LIB.hatrix_matrix_add.restype = ctypes.c_void_p

_LIB.hatrix_matrix_multiply.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_LIB.hatrix_matrix_multiply.restype = ctypes.c_void_p

_LIB.hatrix_matrix_multiply_loop_reordered.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_LIB.hatrix_matrix_multiply_loop_reordered.restype = ctypes.c_void_p

_LIB.hatrix_matrix_transpose.argtypes = [ctypes.c_void_p]
_LIB.hatrix_matrix_transpose.restype = ctypes.c_void_p

_LIB.hatrix_matrix_kronecker.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_LIB.hatrix_matrix_kronecker.restype = ctypes.c_void_p

_LIB.hatrix_matrix_normalize.argtypes = [ctypes.c_void_p]
_LIB.hatrix_matrix_normalize.restype = ctypes.c_void_p

_LIB.hatrix_matrix_sylvester.argtypes = [ctypes.c_size_t]
_LIB.hatrix_matrix_sylvester.restype = ctypes.c_void_p

_LIB.hatrix_matrix_is_hadamard.argtypes = [ctypes.c_void_p]
_LIB.hatrix_matrix_is_hadamard.restype = ctypes.c_int

_LIB.hatrix_matrix_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_LIB.hatrix_matrix_save.restype = ctypes.c_int

_LIB.hatrix_matrix_load.argtypes = [ctypes.c_char_p]
_LIB.hatrix_matrix_load.restype = ctypes.c_void_p

_LIB.hatrix_matrix_copy_data.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
_LIB.hatrix_matrix_copy_data.restype = ctypes.c_int

_LIB.hatrix_last_error_message.argtypes = []
_LIB.hatrix_last_error_message.restype = ctypes.c_char_p

_LIB.hatrix_clear_error.argtypes = []
_LIB.hatrix_clear_error.restype = None


def _raise_last_error() -> None:
    message = _LIB.hatrix_last_error_message()
    text = message.decode("utf-8") if message else "unknown error"
    _LIB.hatrix_clear_error()
    raise RuntimeError(text)


def _check_pointer(pointer: int) -> int:
    if not pointer:
        _raise_last_error()
    return pointer


def _check_status(status: int) -> None:
    if status != 1:
        _raise_last_error()


class Matrix:
    def __init__(self, rows: int, cols: int, data: list[float] | None = None):
        if data is None:
            handle = _LIB.hatrix_matrix_create(rows, cols)
        else:
            flat = [float(value) for value in data]
            buffer = (ctypes.c_double * len(flat))(*flat)
            handle = _LIB.hatrix_matrix_create_from_data(rows, cols, buffer, len(flat))
        self._handle = _check_pointer(handle)

    @classmethod
    def _from_handle(cls, handle: int) -> "Matrix":
        matrix = cls.__new__(cls)
        matrix._handle = _check_pointer(handle)
        return matrix

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            _LIB.hatrix_matrix_destroy(handle)
            self._handle = None

    @property
    def rows(self) -> int:
        value = _LIB.hatrix_matrix_rows(self._handle)
        if _LIB.hatrix_last_error_message():
            _raise_last_error()
        return int(value)

    @property
    def cols(self) -> int:
        value = _LIB.hatrix_matrix_cols(self._handle)
        if _LIB.hatrix_last_error_message():
            _raise_last_error()
        return int(value)

    def get(self, row: int, col: int) -> float:
        value = _LIB.hatrix_matrix_get(self._handle, row, col)
        if _LIB.hatrix_last_error_message():
            _raise_last_error()
        return float(value)

    def set(self, row: int, col: int, value: float) -> None:
        _check_status(_LIB.hatrix_matrix_set(self._handle, row, col, value))

    def swap_rows(self, first: int, second: int) -> None:
        _check_status(_LIB.hatrix_matrix_swap_rows(self._handle, first, second))

    def swap_cols(self, first: int, second: int) -> None:
        _check_status(_LIB.hatrix_matrix_swap_cols(self._handle, first, second))

    def negate_row(self, row: int) -> None:
        _check_status(_LIB.hatrix_matrix_negate_row(self._handle, row))

    def negate_col(self, col: int) -> None:
        _check_status(_LIB.hatrix_matrix_negate_col(self._handle, col))

    def add(self, other: "Matrix") -> "Matrix":
        return Matrix._from_handle(_LIB.hatrix_matrix_add(self._handle, other._handle))

    def multiply(self, other: "Matrix") -> "Matrix":
        return Matrix._from_handle(_LIB.hatrix_matrix_multiply(self._handle, other._handle))

    def multiply_loop_reordered(self, other: "Matrix") -> "Matrix":
        return Matrix._from_handle(
            _LIB.hatrix_matrix_multiply_loop_reordered(self._handle, other._handle)
        )

    def transpose(self) -> "Matrix":
        return Matrix._from_handle(_LIB.hatrix_matrix_transpose(self._handle))

    def kronecker(self, other: "Matrix") -> "Matrix":
        return Matrix._from_handle(_LIB.hatrix_matrix_kronecker(self._handle, other._handle))

    def normalize(self) -> "Matrix":
        return Matrix._from_handle(_LIB.hatrix_matrix_normalize(self._handle))

    def is_hadamard(self) -> bool:
        value = _LIB.hatrix_matrix_is_hadamard(self._handle)
        if _LIB.hatrix_last_error_message():
            _raise_last_error()
        return bool(value)

    @classmethod
    def sylvester(cls, power: int) -> "Matrix":
        return cls._from_handle(_LIB.hatrix_matrix_sylvester(power))

    def save(self, path: str) -> None:
        _check_status(_LIB.hatrix_matrix_save(self._handle, str(path).encode("utf-8")))

    @classmethod
    def load(cls, path: str) -> "Matrix":
        return cls._from_handle(_LIB.hatrix_matrix_load(str(path).encode("utf-8")))

    def to_list(self) -> list[list[float]]:
        size = self.rows * self.cols
        buffer = (ctypes.c_double * size)()
        _check_status(_LIB.hatrix_matrix_copy_data(self._handle, buffer, size))
        flat = [float(buffer[index]) for index in range(size)]
        return [
            flat[row * self.cols : (row + 1) * self.cols]
            for row in range(self.rows)
        ]

    def __repr__(self) -> str:
        return f"Matrix(rows={self.rows}, cols={self.cols}, data={self.to_list()})"

    def __len__(self) -> int:
        return self.rows

    def __add__(self, other: "Matrix") -> "Matrix":
        return self.add(other)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        return self.multiply(other)

    def __getitem__(self, key: tuple[int, int]) -> float:
        row, col = key
        return self.get(row, col)

    def __setitem__(self, key: tuple[int, int], value: float) -> None:
        row, col = key
        self.set(row, col, value)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)
