cmake_minimum_required(VERSION 3.20)

project(Hatrix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(hatrix SHARED
    src/CAPI.cpp
    src/Matrix.cpp
)

target_include_directories(hatrix
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(hatrix
    PRIVATE
        HATRIX_BUILDING_LIBRARY
)

set_target_properties(hatrix PROPERTIES
    OUTPUT_NAME hatrix
    POSITION_INDEPENDENT_CODE ON
)

if(UNIX AND NOT APPLE)
    set_target_properties(hatrix PROPERTIES
        BUILD_RPATH "$ORIGIN"
    )
endif()

enable_testing()

add_executable(hatrix_bench
    bench/cpp_bench.cpp
)

target_link_libraries(hatrix_bench
    PRIVATE
        hatrix
)

add_executable(hatrix_cpp_tests
    test/matrix_cpp_tests.cpp
)

target_link_libraries(hatrix_cpp_tests
    PRIVATE
        hatrix
)

add_executable(hatrix_capi_tests
    test/capi_cpp_tests.cpp
)

target_link_libraries(hatrix_capi_tests
    PRIVATE
        hatrix
)

add_test(NAME hatrix_cpp_tests COMMAND hatrix_cpp_tests)
add_test(NAME hatrix_capi_tests COMMAND hatrix_capi_tests)
add_test(
    NAME hatrix_python_tests
    COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python
        ${Python3_EXECUTABLE} -m unittest -v test_hatrix
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/python
)
