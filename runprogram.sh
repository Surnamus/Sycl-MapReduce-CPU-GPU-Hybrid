#!/bin/bash
set -euo pipefail

# pad_array utility
file3="/home/user/project/points.txt"

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

  for k in "${Karr[@]}"; do
    local N="$N_start"
    local Ncount="$NumberOfNs"
    truncate -s 0 "$file3"
    while [ "$Ncount" -gt 0 ]; do
      local idx=$(( NumberOfNs - Ncount ))
      local localsize_elem="${localsizearr[$idx]}"
      local localsizecpu_elem="${localsizearrcpuhyb[$idx]}"

      for device in 1 2 3; do
        echo "Running with N=$N, K=$k, LS=$localsize_elem, BS=$localsizecpu_elem, dev=$device..."

        ./execute.sh "$N" "$k" "$localsize_elem" "$localsizecpu_elem" "$device"

        val=$(./scripts/measure.sh "$N" "$k" "$localsize_elem" "$localsizecpu_elem" "$device" "$metric")

        printf "%s %s %s %s %s %s %s\n" \
          "$N" "$k" "$localsize_elem" "$localsizecpu_elem" "$device" "$metric" "$val" >> "$POINTS_FILE"
      done

      N=$(( N + increment ))
      Ncount=$(( Ncount - 1 ))
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

    # Convert the new, expanded arrays into space-separated strings for the plotter
    localsize_elem_str="${expanded_ls[*]}"
    localsizecpu_elem_str="${expanded_lscpu[*]}"

    ./scripts/paramplotter.py "$POINTS_FILE" "$metric" "$increment" "$k" "$localsize_elem_str" "$localsizecpu_elem_str"
    else
    python3 ./scripts/paramplotter.py "$POINTS_FILE" "$metric" "$increment" "$k" "$localsize_elem" "$localsizecpu_elem"
    fi
  done

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


run_experiments "$N_start" "$increment" "$NumberOfNs" "$metric" Karr[@] localsizearr[@] localsizearrcpuhyb[@]

run_experiments "$N_start" 0 "${#bs[@]}" "$metric" Karr[@] bs[@] bsc[@]
