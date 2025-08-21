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
    return int(sys.argv[5])
    
def plotter2par(measx, measy, xlabel, ylabel, savedir, prefix=None):
    if not measx or not measy:
        return
    fig = plt.figure()
    plt.plot(measx, measy, color=randcolor())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safepref = prefix if prefix else f"{xlabel}_vs_{ylabel}"
    os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, f"{safepref}_{ts}.png")
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {savepath}")

time = []
gtemp = []
ctemp = []
gutil = []
cutil = []
gmem = []
cmem = []

file_path = 'measurements.log'

try:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    time_indexes = [i for i, line in enumerate(lines) if line.startswith("Time(s)")]
    if not time_indexes:
        raise ValueError("No 'Time(s)' found in file.")
    start_idx = time_indexes[-1] + 1
    for line in lines[start_idx:]:
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

    project_root = "project"
    device_folder = detect_device_type()
    if device_folder not in ["CPU","GPU","Hybrid"]:
        device_folder = "CPU"

    for sub in ["CPU","GPU","Hybrid"]:
        os.makedirs(os.path.join(project_root, "results", sub), exist_ok=True)

    savedir = os.path.join(project_root, "results", device_folder)
    plotter2par(time, cutil, 'time', 'cpu_util', savedir, "time_cpu_util")
    plotter2par(time, gutil, 'time', 'gpu_util', savedir, "time_gpu_util")
    plotter2par(time, ctemp, 'time', 'cpu_temp', savedir, "time_cpu_temp")
    plotter2par(time, gtemp, 'time', 'gpu_temp', savedir, "time_gpu_temp")
    plotter2par(gutil, gtemp, 'gpu_util', 'gpu_temp', savedir, "gpu_util_gpu_temp")
    plotter2par(cutil, ctemp, 'cpu_util', 'cpu_temp', savedir, "cpu_util_cpu_temp")
    plotter2par(gtemp, gmem, 'gpu_temp', 'gpu_mem', savedir, "gpu_temp_gpu_mem")
    plotter2par(ctemp, cmem, 'cpu_temp', 'cpu_mem', savedir, "cpu_temp_cpu_mem")
    plotter2par(gutil, gmem, 'gpu_util', 'gpu_mem', savedir, "gpu_util_gpu_mem")
    plotter2par(cutil, cmem, 'cpu_util', 'cpu_mem', savedir, "cpu_util_cpu_mem")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
