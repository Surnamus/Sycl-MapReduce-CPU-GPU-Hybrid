#!/bin/bash
set -e  # Stop immediately if any command fails

mkdir -p build
cd build

rm -rf ./*

cmake -DCMAKE_BUILD_TYPE=Release ..

cmake --build . --parallel 4
wait   
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
