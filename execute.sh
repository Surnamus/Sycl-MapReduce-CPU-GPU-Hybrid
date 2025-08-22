#!/bin/bash
set -e  # Stop immediately if any command fails
N="$1"
K="$2"
LS="$3"
BS="$4"
dev="$5"
mkdir -p build
cd build

rm -rf ./*

cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel 4
cd ..
./scripts/outcleaner.sh < /dev/null
./scripts/decompressor.sh < /dev/null
./scripts/modifier.sh < /dev/null
./scripts/truncator.sh  "$N"  < /dev/null
python3 ./scripts/solutiongenerator.py "$N" "$K" 
#./scripts/testsuite.sh "$N" "$K" "$LS" "$BS" "$dev"
#./scripts/measure.sh "$N" "$K" "$LS" "$BS" "$dev" "$met"


#python3 ./scripts/plotter.py "$N" "$K" "$LS" "$BS" "$dev"
