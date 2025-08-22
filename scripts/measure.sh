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
#start_time=$(date +%s)
# Run the program and redirect its stdout/stderr to the log to keep our output clean
    
#device="$dev" "/home/user/project/build/project" "$N" "$K" "$LS" "$BS" "$dev" >> "$LOGFILE" 2>&1 || true

#exec_time=$((end_time - start_time))
exec_time=$(
    { /usr/bin/time -f "%e" env device="$dev" /home/user/project/build/project "$N" "$K" "$LS" "$BS" "$dev"; } 2>&1 \
    | tee -a "$LOGFILE" \
    | tail -n 1 | awk '{printf "%.3f", $1*1000}'
)

end_time=$(date +%s)

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

# --- Full Log EnAtry ---
printf "exec_time=%s cpu_temp=%s gpu_temp=%s gpu_util=%s gpu_mem=%s cpu_util=%s cpu_mem=%s\n" \
    "$exec_time" "$cpu_temp" "$gpu_temp" "$gpu_util" "$gpu_mem" "$cpu_util" "$cpu_mem" >> "$LOGFILE"

# --- Prepare and Output the Final Metric ---
metrics=( "$exec_time" "$gpu_util" "$cpu_util" "$gpu_mem" "$cpu_mem" "$gpu_temp" "$cpu_temp" )
METRIC="${metrics[$met]}"

# This is the ONLY line that should print to standard output.
printf "%s\n" "$METRIC"