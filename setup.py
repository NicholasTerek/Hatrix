# Hatrix
# Copyright (c) 2025 Hatrix contributors
# Licensed under the MIT License. See LICENSE for details.

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "python" / "hatrix"
BUILD_DIR = ROOT / "build_python"


def _library_name() -> str:
    if sys.platform == "win32":
        return "hatrix.dll"
    if sys.platform == "darwin":
        return "libhatrix.dylib"
    return "libhatrix.so"


def _build_shared_library() -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["cmake", "-S", str(ROOT), "-B", str(BUILD_DIR)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        ["cmake", "--build", str(BUILD_DIR)],
        check=True,
        cwd=ROOT,
    )

    built_library = BUILD_DIR / _library_name()
    if not built_library.exists():
        raise RuntimeError(f"expected built library at {built_library}")

    shutil.copy2(built_library, PACKAGE_DIR / _library_name())


class build_py(_build_py):
    def run(self) -> None:
        _build_shared_library()
        super().run()


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


setup(cmdclass={"build_py": build_py}, distclass=BinaryDistribution)
