#!/bin/bash

LOGFILE="measurements.log"
INTERVAL=0.1

echo "Start time: $(date)" >> "$LOGFILE"

echo >> "$LOGFILE"
echo "CPU (start):" >> "$LOGFILE"
sensors | grep -E 'Core|Package' >> "$LOGFILE"

echo >> "$LOGFILE"
echo "GPU (start):" >> "$LOGFILE"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits | tr -d ',' >> "$LOGFILE"

echo >> "$LOGFILE"
echo "Running ./main..." >> "$LOGFILE"
start=$(date +%s)

/home/user/project/build/project & PID=$!

while [ ! -f start_measure ]; do
    sleep 0.05
done
rm -f start_measure
echo >> "$LOGFILE"
echo "Time(s) CPU_Temp GPU_Temp GPU_Util GPU_Mem CPU_Util CPU_Mem" >> "$LOGFILE"

while kill -0 "$PID" 2>/dev/null; do
    now=$(( $(date +%s) - start ))

    cpu_temp=$(sensors | grep 'Tctl' | awk '{print $2}' | sed 's/+//' | sed 's/°C//')
    cpu_temp=${cpu_temp:-0}

    gpu_stats=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ',' | grep -E '^[0-9]')
    if [[ -n "$gpu_stats" ]]; then
        read -r gpu_temp gpu_util gpu_mem_used <<< "$gpu_stats"
    else
        gpu_temp=0
        gpu_util=0
        gpu_mem_used=0
    fi

    # CPU utilization
    cpu_util=$(top -bn1 | awk '/Cpu\(s\)/ {print 100 - $8}')
    cpu_util=${cpu_util:-0}

    # Memory usage
    mem_used=$(free -m | awk '/Mem:/ {print $3}')
    mem_used=${mem_used:-0}

    printf "%d %s %s %s %s %.1f %s\n" \
        "$now" "$cpu_temp" "$gpu_temp" "$gpu_util" "$gpu_mem_used" "$cpu_util" "$mem_used" \
        >> "$LOGFILE"

    sleep "$INTERVAL"
done

end=$(date +%s)

echo >> "$LOGFILE"
echo "End time: $(date)" >> "$LOGFILE"
echo "Execution time: $((end - start)) seconds" >> "$LOGFILE"

echo >> "$LOGFILE"
echo "CPU (end):" >> "$LOGFILE"
sensors | grep -E 'Core|Package' >> "$LOGFILE"

echo >> "$LOGFILE"
echo "GPU (end):" >> "$LOGFILE"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits | tr -d ',' >> "$LOGFILE"
