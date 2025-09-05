#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import time
import subprocess
import tempfile
import shutil
from decimal import Decimal, getcontext
from typing import Tuple, Optional

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


def start_nvidia_logger(nvlog_path: str, poll_ms: int = 100) -> Optional[subprocess.Popen]:
    if shutil.which('nvidia-smi') is None:
        return None
    cmd = [
        'nvidia-smi',
        '--query-gpu=temperature.gpu,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits',
        '-lms', str(poll_ms)
    ]
    try:
        f = open(nvlog_path, 'w')
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
        return p
    except Exception:
        return None


def nvlog_last_line(nvlog_path: str) -> str:
    if not os.path.exists(nvlog_path):
        return ''
    try:
        with open(nvlog_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            if pos == 0:
                return ''
            line = b''
            while pos > 0:
                pos -= 1
                f.seek(pos)
                ch = f.read(1)
                if ch == b'\n' and line:
                    break
                if ch != b'\n' and ch != b'\r':
                    line = ch + line
            try:
                text = line.decode('utf-8', errors='ignore').strip()
                return text
            except Exception:
                return ''
    except Exception:
        return ''


def launch_program(path: str, args: list[str], device_env: str) -> subprocess.Popen:
    base_cmd = [path] + args
    if shutil.which('stdbuf'):
        cmd = ['stdbuf', '-oL', '-eL'] + base_cmd
    else:
        cmd = base_cmd
    env = os.environ.copy()
    env['device'] = device_env
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=1, text=True)
    return p


def measure_from_process(proc: subprocess.Popen, nvlog_path: Optional[str], met_index: int):
    vals: list[str] = []
    bg_running = True

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

        gpu_temp = 0
        gpu_util = 0
        gpu_mem = 0
        if nvlog_path and os.path.exists(nvlog_path):
            line = nvlog_last_line(nvlog_path)
            if line:
                line_clean = ''.join(line.split())
                parts = line_clean.split(',')
                if len(parts) >= 3:
                    try:
                        gpu_temp = int(parts[0])
                    except Exception:
                        gpu_temp = 0
                    try:
                        gpu_util = int(parts[1])
                    except Exception:
                        gpu_util = 0
                    try:
                        gpu_mem = int(parts[2])
                    except Exception:
                        gpu_mem = 0
        else:
            if shutil.which('nvidia-smi'):
                try:
                    out = subprocess.check_output([
                        'nvidia-smi',
                        '--query-gpu=temperature.gpu,utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'
                    ], stderr=subprocess.DEVNULL, text=True)
                    first = out.splitlines()[0].strip()
                    parts = [p.strip() for p in first.split(',')]
                    if len(parts) >= 3:
                        gpu_temp = int(parts[0])
                        gpu_util = int(parts[1])
                        gpu_mem = int(parts[2])
                except Exception:
                    gpu_temp = gpu_util = gpu_mem = 0

        metrics = [f"{exec_time_ms:.17f}", str(gpu_util), str(cpu_util), str(gpu_mem), str(cpu_mem), str(gpu_temp), str(cpu_temp)]
        return metrics[met_index]

    while True:
        val = measure_phase_blocking()
        if val is None:
            break
        vals.append(val)
        if proc.poll() is not None and proc.stdout is None:
            break
    return vals


tmp_dir = tempfile.mkdtemp(prefix='measure_py_')
nvlog_path = os.path.join(tmp_dir, 'nvlog.txt')
nv_proc = start_nvidia_logger(nvlog_path, poll_ms=100)

binary = '/home/user/project/build/project'
prog_args = [N, K, LS, BS, dev]
try:
    prog = launch_program(binary, prog_args, dev)
except FileNotFoundError:
    print(f"Error: program not found: {binary}", file=sys.stderr)
    if nv_proc:
        nv_proc.terminate()
    sys.exit(1)

try:
    vals = measure_from_process(prog, nvlog_path if nv_proc else None, met_i)
finally:
    if nv_proc:
        try:
            nv_proc.terminate()
        except Exception:
            pass
    try:
        prog.wait(timeout=1)
    except Exception:
        try:
            prog.terminate()
        except Exception:
            pass
    try:
        shutil.rmtree(tmp_dir)
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
