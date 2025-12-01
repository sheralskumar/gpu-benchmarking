cmake_minimum_required(VERSION 3.10)
project(opencl_bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Let CMake find OpenCL automatically
find_package(OpenCL REQUIRED)

add_executable(benchmark src/benchmark.cpp)

# Include OpenCL headers
target_include_directories(benchmark PRIVATE ${OpenCL_INCLUDE_DIRS})

# Link OpenCL library
target_link_libraries(benchmark PRIVATE ${OpenCL_LIBRARIES})
