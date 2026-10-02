cmake_minimum_required(VERSION 3.20)
project(VectorCore VERSION 0.1.0 LANGUAGES CXX)

option(VECTORCORE_BUILD_TESTS "Build C++ and Python tests" ON)
option(VECTORCORE_BUILD_PYTHON "Build pybind11 Python extension" ON)
option(VECTORCORE_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(VECTORCORE_ENABLE_METAL "Enable Apple Metal backend" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(VECTORCORE_SOURCES
    src/autograd.cpp
    src/nn.cpp
    src/ops.cpp
    src/shape.cpp
    src/storage.cpp
    src/tensor.cpp
    src/cpu/cpu_backend.cpp)

if(APPLE AND VECTORCORE_ENABLE_METAL)
    enable_language(OBJCXX)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/metal")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
    configure_file(metal/kernels.metal "${CMAKE_CURRENT_BINARY_DIR}/metal/kernels.metal" COPYONLY)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/metal/kernels.metal" METAL_SOURCE)
    configure_file(cmake/metal_source.in.hpp
                   "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_source.hpp" @ONLY)
    list(APPEND VECTORCORE_SOURCES src/metal/metal_backend.mm)
else()
    list(APPEND VECTORCORE_SOURCES src/metal/metal_backend_stub.cpp)
endif()

add_library(vectorcore STATIC ${VECTORCORE_SOURCES})
add_library(VectorCore::vectorcore ALIAS vectorcore)
target_include_directories(vectorcore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(vectorcore PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_options(vectorcore PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-fobjc-arc>)

if(APPLE AND VECTORCORE_ENABLE_METAL)
    target_link_libraries(vectorcore PUBLIC "-framework Foundation" "-framework Metal")
    set_source_files_properties(src/metal/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

if(VECTORCORE_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(vectorcore_python MODULE python/bindings.cpp)
    target_link_libraries(vectorcore_python PRIVATE VectorCore::vectorcore)
    set_target_properties(vectorcore_python PROPERTIES OUTPUT_NAME vectorcore)
endif()

if(VECTORCORE_BUILD_TESTS)
    include(CTest)
    add_executable(vectorcore_tests tests/cpp/test_vectorcore.cpp)
    target_link_libraries(vectorcore_tests PRIVATE VectorCore::vectorcore)
    add_test(NAME vectorcore_cpp COMMAND vectorcore_tests)
    if(VECTORCORE_BUILD_PYTHON)
        add_test(NAME vectorcore_python
            COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)
        set_tests_properties(vectorcore_python PROPERTIES
            ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:vectorcore_python>")
    endif()
endif()

add_executable(linear_regression examples/linear_regression.cpp)
target_link_libraries(linear_regression PRIVATE VectorCore::vectorcore)

if(VECTORCORE_BUILD_BENCHMARKS)
    add_executable(vectorcore_bench benchmarks/benchmark_cpp.cpp)
    target_link_libraries(vectorcore_bench PRIVATE VectorCore::vectorcore)
endif()

install(TARGETS vectorcore ARCHIVE DESTINATION lib)
install(DIRECTORY include/vectorcore DESTINATION include)
if(APPLE AND VECTORCORE_ENABLE_METAL)
    install(FILES metal/kernels.metal DESTINATION share/vectorcore/metal)
endif()
