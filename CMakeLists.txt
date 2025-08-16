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