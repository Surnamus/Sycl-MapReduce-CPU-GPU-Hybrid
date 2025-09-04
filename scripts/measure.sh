#!/bin/bash
set -euo pipefail
# usage: ./measure.sh N K LS BS dev met

#FIX SO THAT IT ISNT TOO FAST
N="$1"
K="$2"
LS="$3"
BS="$4"
dev="$5"
met="$6"

# Validate metric index (0=time, 1..6 other metrics)
if ! [[ "$met" =~ ^[0-6]$ ]]; then
    echo "Warning: invalid metric index '$met' - defaulting to 0 (time)" >&2
    met=0
fi

# --- Cleanup trap ---
fifo="/tmp/prog_out.$$"
bg_pid=""
cleanup() {
    exec 3<&- 2>/dev/null || true
    [[ -n "$bg_pid" ]] && kill "$bg_pid" 2>/dev/null || true
    rm -f "$fifo"
}
trap cleanup EXIT

# Wait for a specific string from the program output (line-based, non-busy)
wait_for_signal() {
    local target="$1"
    local line
    while true; do
        # try to read a full line with a short timeout
        if read -r -t 0.2 line <&3 2>/dev/null; then
            # strip possible CR
            line="${line%%$'\r'}"
            if [[ "$line" == *"$target"* ]]; then
                return 0
            fi
            # otherwise continue reading
            continue
        fi

        # no data currently available: check whether the background process is still alive
        if ! kill -0 "$bg_pid" 2>/dev/null; then
            # program exited and we have no more input
            return 1
        fi

        # still running, loop again
    done
}

# One START/STOP pair → one metric value
measure_phase() {
    if ! wait_for_signal "START"; then return 1; fi
    start_time=$(date +%s.%N)
    if ! wait_for_signal "STOP"; then return 1; fi
    end_time=$(date +%s.%N)

    # Exec time in milliseconds, full precision
    exec_time=$(awk -v s="$start_time" -v e="$end_time" 'BEGIN { printf "%.17f", (e - s) * 1000 }')

    # CPU/GPU metrics in full precision (best-effort)
    cpu_temp=$(sensors 2>/dev/null | grep -E 'Tctl|Package id 0|CPU Temp' | head -n1 | awk '{print $2}' | sed 's/+//; s/°C//' || echo "0")
    cpu_temp=${cpu_temp:-0}

    _GPU_RAW=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "")
    if [[ -n "$_GPU_RAW" ]]; then
        IFS=',' read -r raw_gpu_temp raw_gpu_util raw_gpu_mem <<< "$_GPU_RAW"
        gpu_temp=$(echo "$raw_gpu_temp" | tr -d '[:space:]')
        gpu_util=$(echo "$raw_gpu_util" | tr -d '[:space:]')
        gpu_mem=$(echo "$raw_gpu_mem" | tr -d '[:space:]')
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

# --- MAIN ---

# ensure no stale fifo
rm -f "$fifo"
mkfifo "$fifo"

# run program with device env var and line-buffered output so START/STOP aren't stuck in stdio buffers
env device="$dev" stdbuf -oL -eL "/home/user/project/build/project" "$N" "$K" "$LS" "$BS" "$dev" >"$fifo" 2>&1 &
bg_pid=$!

exec 3<"$fifo"
rm -f "$fifo"

vals=()
while kill -0 "$bg_pid" 2>/dev/null; do
    if ! val=$(measure_phase); then
        break
    fi
    vals+=( "$val" )
done

wait "$bg_pid" 2>/dev/null || true

if [ "${#vals[@]}" -eq 0 ]; then
    # keep behaviour of returning "0" but print a hint to stderr for debugging
    echo "No measurements collected (program may have exited before producing START/STOP)." >&2
    echo "0"
    exit 0
fi

# Sum all measurements with full precision
TOTAL=$(printf "%s\n" "${vals[@]}" | awk '{ sum += $1 } END { printf "%.17f", sum }')
echo "$TOTAL"
