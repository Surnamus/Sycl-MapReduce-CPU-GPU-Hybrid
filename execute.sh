#!/bin/bash
set -e  # Stop immediately if any command fails

mkdir -p build
cd build

rm -rf ./*

cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel 4
cd ..
./scripts/decompressor.sh < /dev/null
./scripts/modifier.sh < /dev/null
./scripts/truncator.sh < /dev/null
./scripts/outcleaner.sh < /dev/null
python3 ./scripts/solutiongenerator.py
./scripts/testsuite.sh < /dev/null
python3 ./scripts/verifier.py

#python3 ./scripts/plotter.py
