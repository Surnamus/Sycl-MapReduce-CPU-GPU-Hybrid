from datetime import datetime
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import random

matplotlib.use("Agg")

def randcolor():
    randcmapname = random.choice(plt.colormaps())
    return plt.get_cmap(randcmapname)(random.random())

def detect_device_type():
    try:
        dev = int(sys.argv[5])
    except (IndexError, ValueError):
        return "CPU"
    return {1: "GPU", 2: "CPU", 3: "Hybrid"}.get(dev, "CPU")

def plotter2par(measx, measy, xlabel, ylabel, savedir, prefix=None, title=None):
    if not measx or not measy:
        return
    fig = plt.figure()
    plt.plot(measx, measy, color=randcolor())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safepref = prefix if prefix else f"{xlabel}_vs_{ylabel}"
    os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, f"{safepref}_{ts}.png")
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {savepath}")

# log files are in subfolder
file_path = '/home/user/project/logs/measurements.log'
attempts_path = '/home/user/project/logs/attempts.log'

try:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # load attempt headers
    attempt_params = {}
    attempt_numbers = []
    if os.path.exists(attempts_path):
        with open(attempts_path, 'r') as fa:
            for line in fa:
                if line.startswith("Attempt ") and "N=" in line:
                    num = line.split(":")[0].split()[1]
                    attempt_params[num] = line.strip()
                    attempt_numbers.append(num)

    # find all measurement blocks
    time_indexes = [i for i, line in enumerate(lines) if line.startswith("Time(s)")]
    if not time_indexes:
        raise ValueError("No 'Time(s)' found in file.")

    project_root = "project"
    device_folder = detect_device_type()

    for sub in ["CPU","GPU","Hybrid"]:
        os.makedirs(os.path.join(project_root, "results", sub), exist_ok=True)

    # match each measurement block with attempt numbers
    for block_idx, start_idx in enumerate(time_indexes):
        attempt_num = attempt_numbers[block_idx] if block_idx < len(attempt_numbers) else str(block_idx+1)

        time, gtemp, ctemp, gutil, cutil, gmem, cmem = [], [], [], [], [], [], []

        for line in lines[start_idx+1:]:
            if not line.strip() or line.startswith("End time:"):
                break
            lis = line.split()
            if len(lis) < 7:
                continue
            time.append(lis[0])
            ctemp.append(lis[1])
            gtemp.append(lis[2])
            gutil.append(lis[3])
            gmem.append(lis[4])
            cutil.append(lis[5])
            cmem.append(lis[6])

        savedir = os.path.join(project_root, "results", device_folder)
        prefix = f"attempt_{attempt_num}"
        title = attempt_params.get(attempt_num, f"Attempt {attempt_num}")

        plotter2par(time, cutil, 'time', 'cpu_util', savedir, f"{prefix}_time_cpu_util", title)
        plotter2par(time, gutil, 'time', 'gpu_util', savedir, f"{prefix}_time_gpu_util", title)
        plotter2par(time, ctemp, 'time', 'cpu_temp', savedir, f"{prefix}_time_cpu_temp", title)
        plotter2par(time, gtemp, 'time', 'gpu_temp', savedir, f"{prefix}_time_gpu_temp", title)
        plotter2par(gutil, gtemp, 'gpu_util', 'gpu_temp', savedir, f"{prefix}_gpu_util_gpu_temp", title)
        plotter2par(cutil, ctemp, 'cpu_util', 'cpu_temp', savedir, f"{prefix}_cpu_util_cpu_temp", title)
        plotter2par(gtemp, gmem, 'gpu_temp', 'gpu_mem', savedir, f"{prefix}_gpu_temp_gpu_mem", title)
        plotter2par(ctemp, cmem, 'cpu_temp', 'cpu_mem', savedir, f"{prefix}_cpu_temp_cpu_mem", title)
        plotter2par(gutil, gmem, 'gpu_util', 'gpu_mem', savedir, f"{prefix}_gpu_util_gpu_mem", title)
        plotter2par(cutil, cmem, 'cpu_util', 'cpu_mem', savedir, f"{prefix}_cpu_util_cpu_mem", title)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
