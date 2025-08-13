#!/bin/bash

LOGFILE="measurements.log"
INTERVAL=0.1

echo "Start time: $(date)" | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "CPU (start):" | tee -a "$LOGFILE"
sensors | grep -E 'Core|Package' | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "GPU (start):" | tee -a "$LOGFILE"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "Running ./main..." | tee -a "$LOGFILE"
start=$(date +%s)

./main &
PID=$!

echo | tee -a "$LOGFILE"
echo "Time(s)  CPU_Temp(°C)  GPU_Temp(°C)  GPU_Util(%)  GPU_Mem(MB)  CPU_Util(%)  CPU_Mem(MB)" | tee -a "$LOGFILE"

while kill -0 "$PID" 2>/dev/null; do
    now=$(( $(date +%s) - start ))

    # CPU temp
    cpu_temp=$(sensors | awk '/Package id 0/ {print $4}' | tr -d '+°C')

    # GPU stats (only used memory now)
    read -r gpu_temp gpu_power gpu_util gpu_mem_used <<< \
        $(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used \
            --format=csv,noheader,nounits)

    # CPU utilization
    cpu_util=$(top -bn1 | awk '/Cpu\(s\)/ {print 100 - $8}')

    # Memory usage (only used MB now)
    mem_used=$(free -m | awk '/Mem:/ {print $3}')

    # Output in desired order
    printf "%7s  %12s  %12s  %9s  %11s  %10.1f  %11s\n" \
        "$now" "$cpu_temp" "$gpu_temp" "$gpu_util" "$gpu_mem_used" "$cpu_util" "$mem_used" \
        | tee -a "$LOGFILE"

    sleep "$INTERVAL"
done

end=$(date +%s)

echo | tee -a "$LOGFILE"
echo "End time: $(date)" | tee -a "$LOGFILE"
echo "Execution time: $((end - start)) seconds" | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "CPU (end):" | tee -a "$LOGFILE"
sensors | grep -E 'Core|Package' | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "GPU (end):" | tee -a "$LOGFILE"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits | tee -a "$LOGFILE"
