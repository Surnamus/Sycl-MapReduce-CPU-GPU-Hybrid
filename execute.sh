#!/bin/bash
set -e  # Stop immediately if any command fails

mkdir -p build
cd build

rm -rf ./*

cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel 4
cd ..

# Redirect stdin to avoid blocking C++ input later
./scripts/decompressor.sh < /dev/null
./scripts/modifier.sh < /dev/null
./scripts/truncator.sh < /dev/null
./scripts/testsuite.sh < /dev/null

python3 ./scripts/plotter.py
