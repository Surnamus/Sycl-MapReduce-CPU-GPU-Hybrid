#!/bin/bash

LOGFILE="measurements.log"
INTERVAL=0.1

echo "Start time: $(date)" | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "CPU (start):" | tee -a "$LOGFILE"
sensors | grep -E 'Core|Package' | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "GPU (start):" | tee -a "$LOGFILE"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "Running ./main..." | tee -a "$LOGFILE"
start=$(date +%s)

./src/main &
PID=$!

echo | tee -a "$LOGFILE"
echo "Time(s)  CPU_Temp(°C)  GPU_Temp(°C)  GPU_Util(%)  GPU_Mem(MB/MB)" | tee -a "$LOGFILE"

while kill -0 "$PID" 2>/dev/null; do
    now=$(( $(date +%s) - start ))

    cpu_temp=$(sensors | awk '/Package id 0/ {print $4}' | tr -d '+°C')

    read -r gpu_temp gpu_power gpu_util gpu_mem_used gpu_mem_total <<< \
        $(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits)

    printf "%7s  %12s  %12s  %9s  %7s/%s\n" \
        "$now" "$cpu_temp" "$gpu_temp" "$gpu_util" "$gpu_mem_used" "$gpu_mem_total" | tee -a "$LOGFILE"

    sleep "$INTERVAL"
done

end=$(date +%s)

echo | tee -a "$LOGFILE"
echo "End time: $(date)" | tee -a "$LOGFILE"
echo "execution time: $((end - start)) seconds" | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "CPU (end):" | tee -a "$LOGFILE"
sensors | grep -E 'Core|Package' | tee -a "$LOGFILE"

echo | tee -a "$LOGFILE"
echo "GPU (end):" | tee -a "$LOGFILE"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tee -a "$LOGFILE"

