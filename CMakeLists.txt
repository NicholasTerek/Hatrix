#======================================================================
# CMakeLists.txt
#----------------------------------------------------------------------
# CMake build system for the Hadamard matrix library.
# Supports building tests, benchmarks, and examples.
#
# LICENSE: MIT
# AUTHOR : Nicholas Terek
# VERSION: 1.0.0
#======================================================================

cmake_minimum_required(VERSION 3.12)
project(Hatrix VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable SIMD intrinsics
if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma)
    # Check for AVX-512 support
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    if(COMPILER_SUPPORTS_AVX512F)
        add_compile_options(-mavx512f)
    endif()
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

# Build type specific flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
    if(MSVC)
        add_compile_options(/Od /Zi)
    else()
        add_compile_options(-g -O0)
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2 /Oi /Ot /Oy /GL)
        add_link_options(/LTCG)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

#----------------------------------------------------------------------
# Header-only library setup
#----------------------------------------------------------------------
add_library(hatrix INTERFACE)
target_include_directories(hatrix INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Hatrix>
    $<INSTALL_INTERFACE:include>
)

# Set target properties
set_target_properties(hatrix PROPERTIES
    VERSION ${PROJECT_VERSION}
    INTERFACE_hatrix_VERSION ${PROJECT_VERSION}
)

# Add performance libraries if available
find_package(Threads REQUIRED)
target_link_libraries(hatrix INTERFACE Threads::Threads)

# Add SIMD support
if(NOT MSVC)
    target_compile_options(hatrix INTERFACE -mavx2 -mfma)
    if(COMPILER_SUPPORTS_AVX512F)
        target_compile_options(hatrix INTERFACE -mavx512f)
    endif()
else()
    target_compile_options(hatrix INTERFACE /arch:AVX2)
endif()

#----------------------------------------------------------------------
# Google Test
#----------------------------------------------------------------------
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Fetch Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
    
    # Add Google Test executable
    add_executable(test_hadamard_gtest test/test_hadamard_gtest.cpp)
    target_link_libraries(test_hadamard_gtest hatrix gtest_main)
    
    # Add test
    add_test(NAME hadamard_gtest COMMAND test_hadamard_gtest)
    
    # Set test properties
    set_tests_properties(hadamard_gtest PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    # Include Google Test in project
    include(GoogleTest)
    gtest_discover_tests(test_hadamard_gtest)
endif()

#----------------------------------------------------------------------
# Google Benchmark
#----------------------------------------------------------------------
option(BUILD_BENCHMARKS "Build benchmarks" ON)

if(BUILD_BENCHMARKS)
    # Fetch Google Benchmark
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    FetchContent_MakeAvailable(benchmark)
    
    # Add Google Benchmark executable
    add_executable(benchmark_hadamard_gbench bench/benchmark_hadamard_gbench.cpp)
    target_link_libraries(benchmark_hadamard_gbench hatrix benchmark::benchmark benchmark::benchmark_main)
    
    # Add performance benchmark executable
    add_executable(performance_benchmark bench/performance_benchmark.cpp)
    target_link_libraries(performance_benchmark hatrix benchmark::benchmark benchmark::benchmark_main)
endif()

#----------------------------------------------------------------------
# Examples
#----------------------------------------------------------------------
option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    add_executable(basic_usage examples/basic_usage.cpp)
    target_link_libraries(basic_usage hatrix)
    
    add_executable(advanced_usage examples/advanced_usage.cpp)
    target_link_libraries(advanced_usage hatrix)
endif()

#----------------------------------------------------------------------
# Python Bindings (pybind11)
#----------------------------------------------------------------------
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Fetch pybind11
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    
    # Create Python module
    pybind11_add_module(hatrix bindings/python_bindings.cpp)
    target_link_libraries(hatrix PRIVATE hatrix)
    
    # Set module properties
    set_target_properties(hatrix PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        VISIBILITY_INLINES_HIDDEN ON
        OUTPUT_NAME "hatrix"
    )
    
    # Install Python module
    install(TARGETS hatrix
        COMPONENT python
        LIBRARY DESTINATION ${Python3_SITEARCH}
    )
    
    # Install Python examples
    install(FILES 
        examples/python_basic_usage.py
        examples/python_advanced_usage.py
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/python
        COMPONENT python
    )
endif()

#----------------------------------------------------------------------
# Installation
#----------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS hatrix
    EXPORT hatrixTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES Hatrix/hadamard_matrix.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT hatrixTargets
    FILE hatrixTargets.cmake
    NAMESPACE hatrix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hatrix
)

# Create config file
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    hatrixConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hatrixConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hatrixConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hatrix
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hatrixConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/hatrixConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hatrix
)

#----------------------------------------------------------------------
# Packaging
#----------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "hatrix")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Header-only C++17 library for Hadamard matrices and Fast Walsh-Hadamard Transform")
set(CPACK_PACKAGE_VENDOR "Nicholas Terek")
set(CPACK_PACKAGE_CONTACT "nicholas.terek@example.com")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack)

#----------------------------------------------------------------------
# Documentation
#----------------------------------------------------------------------
option(BUILD_DOCS "Build documentation" OFF)

if(BUILD_DOCS)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
            DESTINATION ${CMAKE_INSTALL_DOCDIR}
            OPTIONAL
        )
    else()
        message(WARNING "Doxygen not found. Documentation will not be built.")
    endif()
endif()

#----------------------------------------------------------------------
# Print configuration summary
#----------------------------------------------------------------------
message(STATUS "")
message(STATUS "Hatrix Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build documentation: ${BUILD_DOCS}")
message(STATUS "")
