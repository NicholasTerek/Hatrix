#!/usr/bin/env python3
"""
Setup script for the Hatrix Python package.
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import sys

# Get the directory containing this setup.py
here = os.path.abspath(os.path.dirname(__file__))

# Read the README file
with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "hatrix",
        [
            "bindings/python_bindings.cpp",
        ],
        include_dirs=[
            "Hatrix",
            pybind11.get_include(),
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"1.0.0"')],
    ),
]

setup(
    name="hatrix",
    version="1.0.0",
    author="Nicholas Terek",
    author_email="nicholas.terek@example.com",
    description="Header-only C++17 library for Hadamard matrices and Fast Walsh-Hadamard Transform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hatrix",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="hadamard matrix walsh transform signal processing mathematics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hatrix/issues",
        "Source": "https://github.com/yourusername/hatrix",
        "Documentation": "https://github.com/yourusername/hatrix#readme",
    },
)
