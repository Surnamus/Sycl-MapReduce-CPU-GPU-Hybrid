#!/bin/bash
# --- CONFIGURATION ---
BUILD_DIR=debug
TARGET=project          # name of executable
USE_GPU=1            # 1=GPU (CUDA backend), 0=CPU-only
set -e 
# --- CREATE BUILD DIRECTORY ---
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# --- RUN CMAKE WITH DEBUG FLAGS ---
cmake -DCMAKE_BUILD_TYPE=Debug ..
# Debug flags ensure -g -O0
make -j$(nproc)

# --- RUN DEBUGGER ---
if [ $USE_GPU -eq 1 ]; then
    echo "Running CUDA-GDB for GPU kernel debugging..."
    cuda-gdb ./$TARGET
else
    echo "Running GDB for CPU debugging..."
    gdb ./$TARGET
fi
cd ..
./scripts/decompressor.sh
wait   
./scripts/modifier.sh
wait   
./scripts/truncator.sh
wait   
./scripts/testsuite.sh
wait   
python3 ./scripts/plotter.py
