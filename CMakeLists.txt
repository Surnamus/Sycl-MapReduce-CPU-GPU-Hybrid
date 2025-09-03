cmake_minimum_required(VERSION 3.22)
project(project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER acpp)      # AdaptiveCpp

# Automatically collect all .cpp files in src/
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include multiple header folders automatically
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/GPU_Module
    ${PROJECT_SOURCE_DIR}/src/CPU_Module
    ${PROJECT_SOURCE_DIR}/src/Hybrid_CPU_GPU
)

set(ENV{ACPP_ADAPTIVITY_LEVEL} "0")

message(STATUS "AdaptiveCpp JIT optimizations disabled (ACPP_ADAPTIVITY_LEVEL=0)")

target_compile_options(${PROJECT_NAME} PRIVATE
    -O3                 # aggressive optimizations
    -fstrict-aliasing   # allow strict aliasing assumptions
    -funroll-loops      # loop unrolling
    -ftree-vectorize    # enable auto-vectorization
    -fopenmp-simd       # allow OpenMP SIMD optimizations
    -fno-math-errno     # avoid math() setting errno overhead
    -ffp-contract=fast  # allow fused multiply-add where safe
    -march=x86-64-v3    # portable modern x86 baseline (adjust if needed)
)