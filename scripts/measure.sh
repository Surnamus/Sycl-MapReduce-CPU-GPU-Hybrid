#!/bin/bash
set -euo pipefail

# --- Arguments ---
N="$1" K="$2" LS="$3" BS="$4" dev="$5" met="$6"

# --- Configuration ---
LOGDIR="$HOME/project/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/measurements.log"

# --- Validation ---
if ! [[ "$met" =~ ^[0-6]$ ]]; then
    echo "Warning: invalid metric index '$met' - defaulting to 0 (time)" >&2
    met=0
fi

# --- Logging (send to stderr or a file, not stdout) ---
echo "----" >> "$LOGFILE"
echo "Run start: $(date) — args: N=$N K=$K LS=$LS BS=$BS dev=$dev met=$met" >> "$LOGFILE"

# --- Main Program Execution ---
# Run the program in the background. It is expected to create 'start_measure' and 'stop' files.
env device="$dev" "/home/user/project/build/project" "$N" "$K" "$LS" "$BS" "$dev" >> "$LOGFILE" 2>&1 &

# --- Wait for start_measure ---
while [ ! -f start_measure ]; do
    sleep 0.001
done
start_time_ns=$(date +%s%N)

# --- Wait for stop ---
while [ ! -f stop ]; do
    sleep 0.001
done
end_time_ns=$(date +%s%N)

# --- High-Precision Time Calculation ---
# Use awk for floating-point arithmetic to get a precise duration in milliseconds.
exec_time=$(awk -v start="$start_time_ns" -v end="$end_time_ns" 'BEGIN { printf "%.3f", (end - start) / 1000000 }')


# --- Metric Collection (with robust error handling) ---

# CPU temp. Redirect errors to prevent them from becoming output.
cpu_temp=$(sensors 2>/dev/null | grep -E 'Tctl|Package id 0|CPU Temp' | head -n1 | awk '{print $2}' | sed 's/+//; s/°C//' || echo "0")
cpu_temp=${cpu_temp:-0}

# GPU stats. Redirect errors.
_GPU_RAW=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "")

if [[ -n "$_GPU_RAW" ]]; then
    IFS=',' read -r raw_gpu_temp raw_gpu_util raw_gpu_mem <<< "$_GPU_RAW"
    gpu_temp=$(echo "$raw_gpu_temp" | tr -d '[:space:]')
    gpu_util=$(echo "$raw_gpu_util" | tr -d '[:space:]' | sed 's/%//')
    gpu_mem=$(echo "$raw_gpu_mem" | tr -d '[:space:]' | sed 's/MiB//; s/ MiB//')
else
    gpu_temp=0
    gpu_util=0
    gpu_mem=0
fi

# CPU utilization. Redirect errors.
cpu_util=$(top -bn1 2>/dev/null | awk -F'id,' '/Cpu\(s\)|%Cpu/ { split($0,a,","); for(i in a) if(a[i] ~ /id/) { sub(/.* /,"",a[i]); print 100 - a[i]; exit } }' || echo "0")
cpu_util=${cpu_util:-0}

# CPU memory. Redirect errors.
cpu_mem=$(free -m 2>/dev/null | awk '/Mem:/ {print $3}' || echo "0")
cpu_mem=${cpu_mem:-0}

# --- Full Log Entry ---
printf "exec_time=%s cpu_temp=%s gpu_temp=%s gpu_util=%s gpu_mem=%s cpu_util=%s cpu_mem=%s\n" \
    "$exec_time" "$cpu_temp" "$gpu_temp" "$gpu_util" "$gpu_mem" "$cpu_util" "$cpu_mem" >> "$LOGFILE"

# --- Prepare and Output the Final Metric ---
metrics=( "$exec_time" "$gpu_util" "$cpu_util" "$gpu_mem" "$cpu_mem" "$gpu_temp" "$cpu_temp" )
METRIC="${metrics[$met]}"

# This is the ONLY line that should print to standard output.
printf "%s\n" "$METRIC"