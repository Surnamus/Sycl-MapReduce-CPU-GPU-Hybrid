#!/bin/bash

echo 'N(s):'
read -a Ns
echo 'K(s):'
read -a Ks
echo 'local_size(s):'
read -a lss
echo 'local_size(s) if hybrid, for cpu:'
read -a lssc
echo 'devices(s) 1-GPU, 2-CPU,3-Hybrid(Map on GPU, Reduce and Sort on CPU):'
read -a dev

for i in "${!Ns[@]}"; do
    export N="${Ns[$i]}"
    export K="${Ks[$i]}"
    export LS="${lss[$i]}"
    export BS="${lssc[$i]}"
    export device="${dev[$i]}"
    ./execute.sh "${Ns[$i]}" "${Ks[$i]}" "${lss[$i]}" "${lssc[$i]}" "${dev[$i]}"
    ./scripts/attemptwriter.sh "${Ns[$i]}" "${Ks[$i]}" "${lss[$i]}" "${lssc[$i]}" "${dev[$i]}" $?
    #determines only if cpp program exited with 0 or 1
done