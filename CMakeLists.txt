cmake_minimum_required(VERSION 3.18...4.2)
project(TensorMIST LANGUAGES C CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# CMake compiles ops.c as C and module.cpp as C++.
pybind11_add_module(_C src/bindings/module.cpp src/backend/ops.c)
target_include_directories(_C PRIVATE include)
target_compile_features(_C PRIVATE c_std_99 cxx_std_17)
install(TARGETS _C LIBRARY DESTINATION tensormist RUNTIME DESTINATION tensormist)