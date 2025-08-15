import matplotlib.pyplot as plt
import random

def randcolor():
    randcmapname = random.choice(plt.colormaps())
    cmap = plt.get_cmap(randcmapname)
    randcmapcolor = cmap(random.random())
    return randcmapcolor

def plotter2par(measx, measy, x, y):
    plt.plot(measx, measy, color=randcolor())
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

time = []
gtemp = []
ctemp = []
gutil = []
cutil = []
gmem = []
cmem = []

file_path = 'measurements.log'

try:
    # Read entire file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find all indexes where "Time(s)" occurs
    time_indexes = [i for i, line in enumerate(lines) if line.startswith("Time(s)")]

    if not time_indexes:
        raise ValueError("No 'Time(s)' found in file.")

    # Take the last occurrence
    start_idx = time_indexes[-1] + 1

    # Read until end or until "End time:" is found
    for line in lines[start_idx:]:
        if not line.strip() or line.startswith("End time:"):
            break
        lis = line.split()
        time.append(lis[0])
        ctemp.append(lis[1])
        gtemp.append(lis[2])
        gutil.append(lis[3])
        gmem.append(lis[4])
        cutil.append(lis[5])
        cmem.append(lis[6])

    # Plotting
    plotter2par(time, ctemp, 'time', 'cpu_util')
    plotter2par(time, gtemp, 'time', 'gpu_util')
    plotter2par(time, ctemp, 'time', 'cpu_temp')
    plotter2par(time, gtemp, 'time', 'gpu_temp')
    plotter2par(gutil, gtemp, 'gpu_util', 'gpu_temp')
    plotter2par(cutil, ctemp, 'cpu_util', 'cpu_temp')
    plotter2par(gtemp, gmem, 'gpu_temp', 'gpu_mem')
    plotter2par(ctemp, cmem, 'cpu_temp', 'cpu_mem')
    plotter2par(gutil, gmem, 'gpu_util', 'gpu_mem')
    plotter2par(cutil, cmem, 'cpu_util', 'cpu_mem')

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
