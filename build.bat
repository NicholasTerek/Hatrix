@echo off
REM======================================================================
REM build.bat
REM----------------------------------------------------------------------
REM Enhanced build script for Windows using CMake, Google Test, 
REM Google Benchmark, and Python bindings.
REM
REM LICENSE: MIT
REM AUTHOR : Nicholas Terek
REM VERSION: 1.0.0
REM======================================================================

echo Building Hatrix library with Google Test, Google Benchmark, and Python bindings...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DBUILD_TESTS=ON ^
    -DBUILD_BENCHMARKS=ON ^
    -DBUILD_EXAMPLES=ON ^
    -DBUILD_PYTHON_BINDINGS=ON

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

REM Run tests
echo.
echo Running Google Test tests...
ctest --config Release --verbose

echo.
echo Running Enterprise Features Test...
.\Release\test_enterprise_features.exe

REM Run benchmarks
echo.
echo Running Google Benchmark...
.\Release\benchmark_hadamard_gbench.exe --benchmark_format=console

echo.
echo Running Performance Benchmark...
.\Release\performance_benchmark.exe --benchmark_format=console

echo.
echo Running Advanced Benchmark...
.\Release\advanced_benchmark.exe --benchmark_format=console

echo.
echo Build completed successfully!
echo.
echo Executables created:
echo   - test_hadamard_gtest.exe (Google Test tests)
echo   - test_performance_gtest.exe (Performance tests)
echo   - benchmark_hadamard_gbench.exe (Google Benchmark)
echo   - performance_benchmark.exe (Performance comparison)
echo   - advanced_benchmark.exe (Advanced GEMM benchmarks)
echo   - basic_usage.exe (basic C++ examples)
echo   - advanced_usage.exe (advanced C++ examples)
echo   - hatrix.pyd (Python module)
echo.
echo To run examples:
echo   .\Release\basic_usage.exe
echo   .\Release\advanced_usage.exe
echo.
echo To run benchmarks:
echo   .\Release\benchmark_hadamard_gbench.exe --benchmark_format=console
echo   .\Release\performance_benchmark.exe --benchmark_format=console
echo   .\Release\advanced_benchmark.exe --benchmark_format=console
echo.
echo To test Python bindings:
echo   python ..\examples\python_basic_usage.py
echo   python ..\examples\python_advanced_usage.py
echo   python ..\examples\python_performance_demo.py
echo.
echo To install Python package:
echo   cd ..
echo   pip install .

pause
