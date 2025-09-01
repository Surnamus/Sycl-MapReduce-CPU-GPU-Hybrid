#!/bin/bash
set -euo pipefail

# usage: ./measure_fix.sh N K LS BS dev met
N="$1" K="$2" LS="$3" BS="$4" dev="$5" met="$6"

# Validate metric index (we keep same ordering; 0=time)
if ! [[ "$met" =~ ^[0-6]$ ]]; then
    echo "Warning: invalid metric index '$met' - defaulting to 0 (time)" >&2
    met=0
fi

# Ensure no stale marker files remain (remove, do not create any new files)
rm -f start_measure stop

# Launch program in background (redirect output to /dev/null so we don't create files)
env device="$dev" "/home/user/project/build/project" "$N" "$K" "$LS" "$BS" "$dev" >/dev/null &
bg_pid=$!

# Helper: wait for one start/stop pair and print a single metric value
measure_phase() {
    # wait for start_measure (exit if bg process died)
    while [ ! -f start_measure ]; do
        sleep 0.001
        if ! kill -0 "$bg_pid" 2>/dev/null; then
            return 1
        fi
    done

    # record start and remove the marker (no other files are created)
    start_time_ns=$(date +%s%N)
    rm -f start_measure

    # wait for stop (exit if bg process died)
    while [ ! -f stop ]; do
        sleep 0.001
        if ! kill -0 "$bg_pid" 2>/dev/null; then
            return 1
        fi
    done

    end_time_ns=$(date +%s%N)
    rm -f stop

    # precise, consistently formatted exec time in milliseconds (3 decimal places)
    exec_time=$(awk -v s="$start_time_ns" -v e="$end_time_ns" 'BEGIN { printf "%.3f", (e - s) / 1000000 }')

    # parse other metrics (best-effort) without creating files
    cpu_temp=$(sensors 2>/dev/null | grep -E 'Tctl|Package id 0|CPU Temp' | head -n1 | awk '{print $2}' | sed 's/+//; s/°C//' || echo "0")
    cpu_temp=${cpu_temp:-0}

    _GPU_RAW=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "")
    if [[ -n "$_GPU_RAW" ]]; then
        IFS=',' read -r raw_gpu_temp raw_gpu_util raw_gpu_mem <<< "$_GPU_RAW"
        gpu_temp=$(echo "$raw_gpu_temp" | tr -d '[:space:]')
        gpu_util=$(echo "$raw_gpu_util" | tr -d '[:space:]' | sed 's/%//')
        gpu_mem=$(echo "$raw_gpu_mem" | tr -d '[:space:]' | sed 's/MiB//; s/ MiB//')
    else
        gpu_temp=0; gpu_util=0; gpu_mem=0
    fi

    cpu_util=$(top -bn1 2>/dev/null | awk -F'id,' '/Cpu\(s\)|%Cpu/ { split($0,a,","); for(i in a) if(a[i] ~ /id/) { sub(/.* /,"",a[i]); print 100 - a[i]; exit } }' || echo "0")
    cpu_util=${cpu_util:-0}

    cpu_mem=$(free -m 2>/dev/null | awk '/Mem:/ {print $3}' || echo "0")
    cpu_mem=${cpu_mem:-0}

    metrics=( "$exec_time" "$gpu_util" "$cpu_util" "$gpu_mem" "$cpu_mem" "$gpu_temp" "$cpu_temp" )
    printf "%s\n" "${metrics[$met]}"
}

# Collect per-phase values in-memory (bash array)
vals=()
while kill -0 "$bg_pid" 2>/dev/null; do
    val=$(measure_phase) || break
    if [[ -n "$val" ]]; then
        vals+=( "$val" )
    else
        break
    fi
done

# If no measurements, print 0
if [ "${#vals[@]}" -eq 0 ]; then
    echo "0"
    exit 0
fi

# Sum measurements using awk reading from STDIN (no files created)
TOTAL=$(printf "%s\n" "${vals[@]}" | awk '{ sum += $1 } END { printf "%.3f", sum }')

echo "$TOTAL"
