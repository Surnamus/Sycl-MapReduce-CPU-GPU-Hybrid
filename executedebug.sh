#!/bin/bash
# --- CONFIGURATION ---
BUILD_DIR=debug
TARGET=project          # name of executable
USE_GPU=1           # 1=GPU (CUDA backend), 0=CPU-only
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
./scripts/outcleaner.sh < /dev/null
./scripts/decompressor.sh < /dev/null
./scripts/modifier.sh < /dev/null
./scripts/truncator.sh  "$N"  < /dev/null
python3 ./scripts/solutiongenerator.py "$N" "$K" "$LS" "$BS" 
./scripts/testsuite.sh "$N" "$K" "$LS" "$BS" "$dev" 
python3 ./scripts/verifier.py

#python3 ./scripts/plotter.py "$dev"
