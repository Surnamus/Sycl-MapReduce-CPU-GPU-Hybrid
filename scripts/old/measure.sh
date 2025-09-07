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
nvlog="/tmp/nvlog.$$"
nv_pid=""
cleanup() {
    exec 3<&- 2>/dev/null || true
    [[ -n "$bg_pid" ]] && kill "$bg_pid" 2>/dev/null || true
    [[ -n "$nv_pid" ]] && kill "$nv_pid" 2>/dev/null || true
    rm -f "$fifo" "$nvlog"
}
trap cleanup EXIT

# --- lightweight helpers (no heavy external commands) ---

# read cpu total and idle from /proc/stat (print "total idle")
read_cpu_times() {
    IFS= read -r line < /proc/stat
    set -- $line
    user=${2:-0}; nice=${3:-0}; system=${4:-0}; idle=${5:-0}; iowait=${6:-0}; irq=${7:-0}; softirq=${8:-0}; steal=${9:-0}
    total=$((user + nice + system + idle + iowait + irq + softirq + steal))
    idle_all=$((idle + iowait))
    printf "%d %d" "$total" "$idle_all"
}

# read first reasonable cpu temp in °C from /sys
read_cpu_temp_c() {
    for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
        typ="/sys/class/thermal/thermal_zone${i}/type"
        tmp="/sys/class/thermal/thermal_zone${i}/temp"
        if [[ -r "$typ" && -r "$tmp" ]]; then
            IFS= read -r ttype < "$typ"
            if echo "$ttype" | grep -qiE 'cpu|x86_pkg_temp' >/dev/null 2>&1 || [[ $i -eq 0 ]]; then
                if IFS= read -r val <"$tmp"; then
                    if (( val > 1000 )); then
                        printf "%d" $((val / 1000))
                    else
                        printf "%d" "$val"
                    fi
                    return 0
                fi
            fi
        fi
    done
    printf "0"
}

# memory used MB via /proc/meminfo (fast)
read_mem_used_mb() {
    mem_total_kb=0; mem_avail_kb=0
    while IFS=':' read -r key val; do
        key=${key// /}
        case "$key" in
            MemTotal) mem_total_kb=$(echo "$val" | tr -dc '0-9') ;;
            MemAvailable) mem_avail_kb=$(echo "$val" | tr -dc '0-9') ;;
        esac
        [[ -n $mem_total_kb && -n $mem_avail_kb && $mem_total_kb -ne 0 ]] && break
    done < /proc/meminfo
    if [[ -z $mem_total_kb || -z $mem_avail_kb || $mem_total_kb -eq 0 ]]; then
        echo 0; return
    fi
    used_kb=$((mem_total_kb - mem_avail_kb))
    echo $((used_kb / 1024))
}

# get last non-empty line from nvlog (fast)
nvlog_last_line() {
    # use tac if available (faster on large files), else use awk
    if command -v tac >/dev/null 2>&1; then
        tac "$nvlog" 2>/dev/null | awk 'NF{print; exit}'
    else
        awk 'NF{line=$0} END{print line}' "$nvlog" 2>/dev/null || true
    fi
}

# --- MAIN ---

# ensure no stale fifo / nvlog
rm -f "$fifo" "$nvlog"
mkfifo "$fifo"

# start background nvidia-smi logger if available (one process for all measurements)
if command -v nvidia-smi >/dev/null 2>&1; then
    # adjust -lms to taste (100 ms is a good default)
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 100 >"$nvlog" 2>/dev/null &
    nv_pid=$!
else
    nv_pid=""
fi

# run program with device env var and line-buffered output so START/STOP aren't stuck in stdio buffers
env device="$dev" stdbuf -oL -eL "/home/user/project/build/project" "$N" "$K" "$LS" "$BS" "$dev" >"$fifo" 2>&1 &
bg_pid=$!

exec 3<"$fifo"
rm -f "$fifo"

# Wait for a specific string from the program output (line-based, responsive)
wait_for_signal() {
    local target="$1"
    local line
    while true; do
        if read -r -t 0.05 line <&3 2>/dev/null; then
            line="${line%%$'\r'}"
            if [[ "$line" == *"$target"* ]]; then
                return 0
            fi
            continue
        fi
        if ! kill -0 "$bg_pid" 2>/dev/null; then
            return 1
        fi
    done
}

# One START/STOP pair → one metric value (kept structure, but cheaper metrics)
measure_phase() {
    # CPU times for delta-based CPU util
    if ! wait_for_signal "START"; then return 1; fi
    start_time=$(date +%s.%N)
    read -r prev_total prev_idle <<< "$(read_cpu_times)"

    if ! wait_for_signal "STOP"; then return 1; fi
    end_time=$(date +%s.%N)

    # Exec time in milliseconds, full precision
    exec_time=$(awk -v s="$start_time" -v e="$end_time" 'BEGIN { printf "%.17f", (e - s) * 1000 }')

    # CPU temp
    cpu_temp=$(read_cpu_temp_c)
    cpu_temp=${cpu_temp:-0}

    # CPU util using /proc/stat delta
    read -r now_total now_idle <<< "$(read_cpu_times)"
    totald=$((now_total - prev_total))
    idled=$((now_idle - prev_idle))
    if (( totald > 0 )); then
        cpu_util=$(( ((totald - idled) * 100) / totald ))
        (( cpu_util < 0 )) && cpu_util=0
    else
        cpu_util=0
    fi

    # CPU mem from /proc
    cpu_mem=$(read_mem_used_mb)
    cpu_mem=${cpu_mem:-0}

    # GPU metrics: take last line from nvlog (single process cost), fallback to single nvidia-smi call if nvlog absent
    gpu_temp=0; gpu_util=0; gpu_mem=0
    if [[ -n "$nv_pid" && -r "$nvlog" ]]; then
        line=$(nvlog_last_line)
        if [[ -n "$line" ]]; then
            # CSV: temp, util, mem
            line="${line// /}"   # strip spaces
            IFS=',' read -r raw_gpu_temp raw_gpu_util raw_gpu_mem <<< "$line"
            gpu_temp="${raw_gpu_temp:-0}"
            gpu_util="${raw_gpu_util:-0}"
            gpu_mem="${raw_gpu_mem:-0}"
        fi
    else
        # rare fallback: single quick nvidia-smi invocation
        _GPU_RAW=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "")
        if [[ -n "$_GPU_RAW" ]]; then
            IFS=',' read -r raw_gpu_temp raw_gpu_util raw_gpu_mem <<< "$_GPU_RAW"
            gpu_temp=$(echo "$raw_gpu_temp" | tr -d '[:space:]')
            gpu_util=$(echo "$raw_gpu_util" | tr -d '[:space:]')
            gpu_mem=$(echo "$raw_gpu_mem" | tr -d '[:space:]')
        else
            gpu_temp=0; gpu_util=0; gpu_mem=0
        fi
    fi

    metrics=( "$exec_time" "$gpu_util" "$cpu_util" "$gpu_mem" "$cpu_mem" "$gpu_temp" "$cpu_temp" )
    printf "%s\n" "${metrics[$met]}"
}

vals=()
while kill -0 "$bg_pid" 2>/dev/null; do
    val=$(measure_phase) || break
    vals+=( "$val" )
done

wait "$bg_pid" 2>/dev/null || true

if [ "${#vals[@]}" -eq 0 ]; then
    echo "No measurements collected (program may have exited before producing START/STOP)." >&2
    echo "0"
    exit 0
fi

# Sum all measurements with full precision (kept same behavior)
TOTAL=$(printf "%s\n" "${vals[@]}" | awk '{ sum += $1 } END { printf "%.17f", sum }')
echo "$TOTAL"
