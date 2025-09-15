#!/bin/bash
set -euo pipefail

# pad_array utility
file3="/home/user/project/points.txt"
RAWFILE="/home/user/project/DATA.txt"

pad_array() {
  local -n arr=$1
  local target_len=$2
  while [ "${#arr[@]}" -lt "$target_len" ]; do
    arr+=("${arr[-1]}")
  done
}

run_experiments() {
  local N_start=$1
  local increment=$2
  local NumberOfNs=$3
  local metric=$4
  shift 4
  local -a Karr=("${!1}")
  local -a localsizearr=("${!2}")
  local -a localsizearrcpuhyb=("${!3}")
  pad_array localsizearr "$NumberOfNs"
  pad_array localsizearrcpuhyb "$NumberOfNs"
  POINTS_FILE="/home/user/project/points.txt"
  truncate -s 0 "$POINTS_FILE"
  #because the program is using JIT compiler, compile for zero in the first run so that compiler
  #optimises itself properly and then actually start using the Nstart
  for k in "${Karr[@]}"; do
    for i in 1 2 3; do
    for device in 1 2 3; do
      echo "first runs, they iterate over K, get rid of JIT warning"
      #big N
      ./execute.sh 5000000 "${k}" "${localsizearr[0]}" "${localsizearrcpuhyb[0]}" "$device"
     /home/user/project/build/project 5000000 "${k}" "${localsizearr[0]}" "${localsizearrcpuhyb[0]}" "$device" "$metric" #>"$fifo" 2>&1 & bg_pid=$!
    done
    done
    done
   file4="/home/user/project/logs/measurements.log"
    truncate -s 0 "$file4"
    for k in "${Karr[@]}"; do
      local N="$N_start"
      local Ncount="$NumberOfNs"
      cat "$file3" >> "$RAWFILE"
      truncate -s 0 "$file3"
    #truncate -s 0 "$file4"  
    while [ "$Ncount" -gt 0 ]; do
      local idx=$(( NumberOfNs - Ncount ))
      local localsize_elem="${localsizearr[$idx]}"
      local localsizecpu_elem="${localsizearrcpuhyb[$idx]}"
      local BINARY=" /home/user/project/build/project"
      for device in 1 2 3; do
        echo "Running with N=$N, K=$k, LS=$localsize_elem, BS=$localsizecpu_elem, dev=$device..."

        ./execute.sh "$N" "$k" "$localsize_elem" "$localsizecpu_elem" "$device"

        /home/user/project/build/project "$N" "$k" "$localsize_elem" "$localsizecpu_elem" "$device" "$metric"
                  #int N,int k, int lls,int llsc,int device,int metric,double value
        #pass that from cpp nad it will work in that way
        #val=$(./scripts/measure.sh "$N" "$k" "$localsize_elem" "$localsizecpu_elem" "$device" "$metric")
      #  printf "%s %s %s %s %s %s %s\n" \
     #     "$N" "$k" "$localsize_elem" "$localsizecpu_elem" "$device" "$metric" "$val" >> "$POINTS_FILE"
     #           printf "%s %s %s %s %s %s %s\n" \
     #     "$N" "$k" "$localsize_elem" "$localsizecpu_elem" "$device" "$metric" "$val" >> "$RAWFILE"

      done

      N=$(( N + increment ))
      Ncount=$(( Ncount - 1 ))
      truncate -s 0 "$file4"  
    done
        if [ "$increment" -eq 0 ]; then
    # Create new empty arrays to hold the repeated values
    local expanded_ls=()
    local expanded_lscpu=()
    local idx=0
    
    # For each local size in the input, add it to the new arrays 3 times (for device 1, 2, 3)
    for ls_val in "${localsizearr[@]}"; do
        lscpu_val="${localsizearrcpuhyb[$idx]}"
        expanded_ls+=("$ls_val" "$ls_val" "$ls_val")
        expanded_lscpu+=("$lscpu_val" "$lscpu_val" "$lscpu_val")
        idx=$((idx + 1))
    done

    localsize_elem_str="${expanded_ls[*]}"
    localsizecpu_elem_str="${expanded_lscpu[*]}"

    ./scripts/paramplotter.py "$POINTS_FILE" "$metric" "$increment" "$k" "$localsize_elem_str" "$localsizecpu_elem_str"
    else
    python3 ./scripts/paramplotter.py "$POINTS_FILE" "$metric" "$increment" "$k" "$localsize_elem" "$localsizecpu_elem"
    fi
    printf "%s"  "----\n" >> "$RAWFILE"
  done
  cat "$POINTS_FILE" >> "$RAWFILE"
  echo "Done. All data points written to: $POINTS_FILE"
  
}

echo 'N start (initial N):'
read N_start
echo 'increment:'
read increment
echo 'NumberOfNs:'
read NumberOfNs

echo 'K(s) (space-separated):'
read -a Karr
echo 'local_size,only one:'
read -a localsizearr
echo 'local_size if hybrid, for cpu,only one:'
read -a localsizearrcpuhyb
echo 'metric (single number, e.g. 0..6):'
read metric

echo "the second case, input the localsizes for gpu (up to 512) :"
read -a bs
echo "the second case, input the localizes for cpu (up to 512), same size as bsc:"
read -a bsc
truncate -s 0 "$file3"

rm -rf build/* && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel 4
printf "BEGIN" >> "$RAWFILE"
run_experiments "$N_start" "$increment" "$NumberOfNs" "$metric" Karr[@] localsizearr[@] localsizearrcpuhyb[@]
printf "....."
run_experiments "$N_start" 0 "${#bs[@]}" "$metric" Karr[@] bs[@] bsc[@]
printf "END">>"$RAWFILE"
    truncate -s 0 "$file3"
    truncate -s 0 "$file4"  
    ./scripts/outcleaner.sh