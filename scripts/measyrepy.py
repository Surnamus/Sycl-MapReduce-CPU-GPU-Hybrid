#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import time
import subprocess
import shutil
from decimal import Decimal, getcontext
from typing import Tuple, Optional

import pynvml

getcontext().prec = 30

USAGE = "usage: ./measure.py N K LS BS dev met"

if len(sys.argv) != 7:
    print(USAGE, file=sys.stderr)
    sys.exit(2)

N, K, LS, BS, dev, met = sys.argv[1:7]
try:
    met_i = int(met)
    if not (0 <= met_i <= 6):
        raise ValueError
except ValueError:
    print(f"Warning: invalid metric index '{met}' - defaulting to 0 (time)", file=sys.stderr)
    met_i = 0


def read_cpu_times() -> Tuple[int, int]:
    """Return (total, idle_all) as integers like the bash helper."""
    with open('/proc/stat', 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    nums = [int(x) for x in parts[1:]]
    while len(nums) < 9:
        nums.append(0)
    user, nice, system, idle, iowait, irq, softirq, steal = nums[:8]
    total = user + nice + system + idle + iowait + irq + softirq + steal
    idle_all = idle + iowait
    return total, idle_all


def read_cpu_temp_c() -> int:
    for i in range(0, 12):
        typ = f"/sys/class/thermal/thermal_zone{i}/type"
        tmp = f"/sys/class/thermal/thermal_zone{i}/temp"
        try:
            with open(typ, 'r') as f:
                ttype = f.read().strip()
            if 'cpu' in ttype.lower() or 'x86_pkg_temp' in ttype.lower() or i == 0:
                with open(tmp, 'r') as f:
                    val = f.read().strip()
                try:
                    iv = int(val)
                    if iv > 1000:
                        return iv // 1000
                    return iv
                except Exception:
                    continue
        except Exception:
            continue
    return 0


def read_mem_used_mb() -> int:
    mem_total_kb = None
    mem_avail_kb = None
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.replace(' ', '')
            if key == 'MemTotal':
                mem_total_kb = int(''.join(ch for ch in val if ch.isdigit()))
            elif key == 'MemAvailable':
                mem_avail_kb = int(''.join(ch for ch in val if ch.isdigit()))
            if mem_total_kb is not None and mem_avail_kb is not None:
                break
    if not mem_total_kb or not mem_avail_kb:
        return 0
    used_kb = mem_total_kb - mem_avail_kb
    return used_kb // 1024


gpu_handle = None
try:
    pynvml.nvmlInit()
    if pynvml.nvmlDeviceGetCount() > 0:
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    gpu_handle = None


def read_gpu_metrics() -> tuple[int, int, int]:
    """Return (gpu_temp, gpu_util, gpu_mem) or zeros if unavailable."""
    if gpu_handle is None:
        return 0, 0, 0
    try:
        temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        return int(temp), int(util.gpu), mem.used // (1024 * 1024)
    except Exception:
        return 0, 0, 0


def launch_program(path: str, args: list[str], device_env: str) -> subprocess.Popen:
    base_cmd = [path] + args
    if shutil.which('stdbuf'):
        cmd = ['stdbuf', '-oL', '-eL'] + base_cmd
    else:
        cmd = base_cmd
    env = os.environ.copy()
    env['device'] = device_env
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        text=True
    )
    return p


def measure_from_process(proc: subprocess.Popen, met_index: int):
    vals: list[str] = []

    def measure_phase_blocking() -> Optional[str]:
        try:
            start_time_ns = None
            prev_total, prev_idle = read_cpu_times()
            while True:
                line = proc.stdout.readline()
                if line == '':
                    return None
                line = line.rstrip('\n').rstrip('\r')
                if 'START' in line:
                    start_time_ns = time.perf_counter_ns()
                    prev_total, prev_idle = read_cpu_times()
                    break
            while True:
                line = proc.stdout.readline()
                if line == '':
                    return None
                line = line.rstrip('\n').rstrip('\r')
                if 'STOP' in line:
                    end_time_ns = time.perf_counter_ns()
                    break
        except Exception:
            return None

        exec_time_ms = (end_time_ns - start_time_ns) / 1_000_000.0

        cpu_temp = read_cpu_temp_c()

        now_total, now_idle = read_cpu_times()
        totald = now_total - prev_total
        idled = now_idle - prev_idle
        if totald > 0:
            cpu_util = int(((totald - idled) * 100) / totald)
            if cpu_util < 0:
                cpu_util = 0
        else:
            cpu_util = 0

        cpu_mem = read_mem_used_mb()

        gpu_temp, gpu_util, gpu_mem = read_gpu_metrics()

        metrics = [
            f"{exec_time_ms:.17f}",
            str(gpu_util),
            str(cpu_util),
            str(gpu_mem),
            str(cpu_mem),
            str(gpu_temp),
            str(cpu_temp)
        ]
        return metrics[met_index]

    while True:
        val = measure_phase_blocking()
        if val is None:
            break
        vals.append(val)
        if proc.poll() is not None and proc.stdout is None:
            break
    return vals


binary = '/home/user/project/build/project'
prog_args = [N, K, LS, BS, dev]
try:
    prog = launch_program(binary, prog_args, dev)
except FileNotFoundError:
    print(f"Error: program not found: {binary}", file=sys.stderr)
    sys.exit(1)

try:
    vals = measure_from_process(prog, met_i)
finally:
    try:
        prog.wait(timeout=1)
    except Exception:
        try:
            prog.terminate()
        except Exception:
            pass
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass

if not vals:
    print("No measurements collected (program may have exited before producing START/STOP).", file=sys.stderr)
    print("0")
    sys.exit(0)

sum_dec = Decimal('0')
for v in vals:
    try:
        sum_dec += Decimal(v)
    except Exception:
        sum_dec += Decimal(str(float(v)))

print(f"{sum_dec:.17f}")
