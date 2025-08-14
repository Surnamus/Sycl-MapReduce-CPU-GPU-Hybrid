#!/bin/bash
set -e  # Stop immediately if any command fails

mkdir -p build
cd build

rm -rf ./*

cmake -DCMAKE_BUILD_TYPE=Release ..

cmake --build . --parallel 4

cd ..
./scripts/decompressor.sh
./scripts/modifier.sh
./scripts/testsuite.sh
python3 ./scripts/plotter.py
