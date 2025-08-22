#!/usr/bin/env python3
from datetime import datetime
import os
import sys
import re
import glob
import matplotlib
import matplotlib.pyplot as plt
import random
import math

matplotlib.use("Agg")

# ----------------- Helper Functions ----------------- #

def randcolor():
    """Returns a visually distinct random color (not too light)."""
    while True:
        color = (random.random(), random.random(), random.random())
        # avoid very light colors
        if sum(color) / 3 < 0.85:
            return color

def detect_device_type():
    """Detects device type from command-line arguments, defaulting to CPU."""
    try:
        dev = int(sys.argv[5])
    except (IndexError, ValueError):
        return "CPU"
    return {1: "GPU", 2: "CPU", 3: "Hybrid"}.get(dev, "CPU")

def safe_float(token):
    """Convert a token to float robustly, return None if impossible."""
    if token is None:
        return None
    t = str(token).strip().replace('%', '').replace('°C', '').replace('°', '').replace(',', '')
    if t == '':
        return None
    try:
        return float(t)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", t)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None

def _meaningful_ylim(values, ylabel_hint=None):
    """Return sensible y-limits with a small margin around min/max."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return (0.0, 1.0)

    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax, rel_tol=1e-9, abs_tol=1e-9):
        margin = 0.05 * abs(vmin) if vmin != 0 else 1.0
        return (vmin - margin, vmax + margin)

    vrange = vmax - vmin
    margin = max(vrange * 0.03, 0.5)
    ymin, ymax = vmin - margin, vmax + margin

    if ylabel_hint:
        ly = ylabel_hint.lower()
        if 'util' in ly:
            ymin = max(0.0, ymin)
            ymax = min(100.0, ymax)
        if 'mem' in ly:
            ymin = max(0.0, ymin)

    return ymin, ymax

def plotter2par(measx, measy, xlabel, ylabel, savedir, prefix=None, title=None):
    """Plot measy vs measx using actual measured points with margins."""
    if measx is None or measy is None:
        return

    xs_raw = [safe_float(x) for x in measx]
    ys_raw = [safe_float(y) for y in measy]

    # keep only numeric x
    pairs = [(x, math.nan if y is None else y) for x, y in zip(xs_raw, ys_raw) if x is not None]
    if not pairs:
        return

    pairs.sort(key=lambda p: p[0])
    xs, ys = zip(*pairs)
    xs, ys = list(xs), list(ys)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = _meaningful_ylim(ys, ylabel_hint=ylabel)

    # Add tiny margins to x for visibility
    xr = xmax - xmin or 1.0
    xmin -= 0.02 * xr
    xmax += 0.02 * xr

    n_points = len(xs)
    if n_points <= 2:
        msize, lw = 7, 2.2
    elif n_points <= 10:
        msize, lw = 5, 1.6
    else:
        msize, lw = 3, 1.0

    dpi_val = 100
    fig_size_inches = 480 / dpi_val
    fig = plt.figure(figsize=(fig_size_inches, fig_size_inches))

    plt.plot(xs, ys, color=randcolor(), marker='o', markersize=msize,
             linewidth=lw, markeredgecolor='black', markeredgewidth=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safepref = prefix if prefix else f"{xlabel}_vs_{ylabel}"
    os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, f"{safepref}_{ts}.png")
    
    fig.savefig(savepath, dpi=dpi_val)
    plt.close(fig)
    print(f"Saved {savepath}")

# ----------------- Main Parsing & Plotting ----------------- #

file_path = '/home/user/project/logs/measurements.log'
attempts_path = '/home/user/project/logs/attempts.log'

try:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\n') for ln in f]

    attempt_params = {}
    attempt_numbers = []
    if os.path.exists(attempts_path):
        with open(attempts_path, 'r', encoding='utf-8', errors='ignore') as fa:
            for line in fa:
                m = re.match(r'^\s*Attempt\s+(\d+)\s*:\s*(.*\bN\s*=\s*\d+.*)$', line)
                if m:
                    num = m.group(1)
                    hdr = m.group(0).strip()
                    dev_m = re.search(r'\bDEV\s*=\s*(\d+)', hdr, re.IGNORECASE)
                    if dev_m:
                        dev_id = int(dev_m.group(1))
                        dev_word = {1: "GPU", 2: "CPU", 3: "Hybrid"}.get(dev_id, f"DEV={dev_id}")
                        hdr += f" ({dev_word})"
                    attempt_params[num] = hdr
                    attempt_numbers.append(num)

    blocks = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        if lines[i].startswith("Start time:"):
            start_idx = i
            j = i + 1
            end_idx = None
            while j < n_lines:
                if lines[j].startswith("End time:"):
                    end_idx = j
                    break
                j += 1
            if end_idx is None:
                end_idx = n_lines - 1
            blocks.append((start_idx, end_idx))
            i = end_idx + 1
        else:
            i += 1

    if not blocks:
        raise ValueError("No Start/End measurement blocks found in measurements.log")

    project_root = "project"
    device_folder = detect_device_type()
    for sub in ["CPU", "GPU", "Hybrid"]:
        os.makedirs(os.path.join(project_root, "results", sub), exist_ok=True)

    for block_idx, (start_idx, end_idx) in enumerate(blocks):
        header_idx = None
        for k in range(start_idx, end_idx + 1):
            if lines[k].startswith("Time(s)"):
                header_idx = k
                break
        if header_idx is None:
            print(f"Block {block_idx+1}: no 'Time(s)' header found, skipping")
            continue

        attempt_num = attempt_numbers[block_idx] if block_idx < len(attempt_numbers) else str(block_idx+1)
        title = attempt_params.get(attempt_num, f"Attempt {attempt_num}")

        # Collect measurement rows
        time_vals, ctemp_vals, gtemp_vals, gutil_vals, gmem_vals, cutil_vals, cmem_vals = [], [], [], [], [], [], []
        for row in lines[header_idx+1:end_idx]:
            if row.startswith("End time:") or not row.strip():
                continue
            parts = row.split()
            if len(parts) < 7:
                continue
            t = safe_float(parts[0])
            if t is None:
                continue
            time_vals.append(t)
            ctemp_vals.append(safe_float(parts[1]))
            gtemp_vals.append(safe_float(parts[2]))
            gutil_vals.append(safe_float(parts[3]))
            gmem_vals.append(safe_float(parts[4]))
            cutil_vals.append(safe_float(parts[5]))
            cmem_vals.append(safe_float(parts[6]))

        if not time_vals:
            print(f"Block {block_idx+1} (Attempt {attempt_num}): no numeric rows found, skipping")
            continue

        savedir = os.path.join(project_root, "results", device_folder)
        prefix = f"attempt_{attempt_num}"
        os.makedirs(savedir, exist_ok=True)

        # Skip re-plotting existing attempts
        existing = glob.glob(os.path.join(savedir, f"{prefix}_*.png"))
        if existing:
            print(f"Attempt {attempt_num}: plots already exist ({len(existing)} files) — skipping")
            continue

        # Generate plots
        plotter2par(time_vals, cutil_vals, 'time (s)', 'cpu_util', savedir, f"{prefix}_time_cpu_util", title)
        plotter2par(time_vals, gutil_vals, 'time (s)', 'gpu_util', savedir, f"{prefix}_time_gpu_util", title)
        plotter2par(time_vals, ctemp_vals, 'time (s)', 'cpu_temp', savedir, f"{prefix}_time_cpu_temp", title)
        plotter2par(time_vals, gtemp_vals, 'time (s)', 'gpu_temp', savedir, f"{prefix}_time_gpu_temp", title)
        plotter2par(gutil_vals, gtemp_vals, 'gpu_util', 'gpu_temp', savedir, f"{prefix}_gpu_util_gpu_temp", title)
        plotter2par(cutil_vals, ctemp_vals, 'cpu_util', 'cpu_temp', savedir, f"{prefix}_cpu_util_cpu_temp", title)
        plotter2par(gtemp_vals, gmem_vals, 'gpu_temp', 'gpu_mem', savedir, f"{prefix}_gpu_temp_gpu_mem", title)
        plotter2par(ctemp_vals, cmem_vals, 'cpu_temp', 'cpu_mem', savedir, f"{prefix}_cpu_temp_cpu_mem", title)
        plotter2par(gutil_vals, gmem_vals, 'gpu_util', 'gpu_mem', savedir, f"{prefix}_gpu_util_gpu_mem", title)
        plotter2par(cutil_vals, cmem_vals, 'cpu_util', 'cpu_mem', savedir, f"{prefix}_cpu_util_cpu_mem", title)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
