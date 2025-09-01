#!/usr/bin/env python3
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# --- Configuration ---
METRIC_NAMES = {
    "0": "Time (ms)",
    "1": "GPU Util (%)",
    "2": "CPU Util (%)",
    "3": "GPU Mem (MB)",
    "4": "CPU Mem (MB)",
    "5": "GPU Temp (°C)",
    "6": "CPU Temp (°C)",
}

def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

def parse_points_file(path):

    data = {'N': [], 'dev': [], 'metric': []}
    
    if not os.path.exists(path):
        print(f"Error: Points file not found: {path}", file=sys.stderr)
        return None

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) < 7:
                print(f"Warning: Skipping malformed line {i+1} in {path}", file=sys.stderr)
                continue

            n_val = safe_float(parts[0])
            dev_val = safe_float(parts[4])
            metric_val = safe_float(parts[6])

            if all(v is not None for v in [n_val, dev_val, metric_val]):
                data['N'].append(n_val)
                data['dev'].append(int(dev_val))
                data['metric'].append(metric_val)
            else:
                print(f"Warning: Could not parse numeric values on line {i+1} in {path}", file=sys.stderr)
    
    return data

def generate_plot(x_data, y_data, dev_list, x_label, y_label, title, metric_name):
    
    savedir = "./results"
    os.makedirs(savedir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    plot_data = {
        1: {'x': [], 'y': [], 'label': 'GPU', 'marker': 'o'},
        2: {'x': [], 'y': [], 'label': 'CPU', 'marker': 's'},
        3: {'x': [], 'y': [], 'label': 'Hybrid', 'marker': '^'}
    }

    for i, dev in enumerate(dev_list):
        if dev in plot_data:
            plot_data[dev]['x'].append(x_data[i])
            plot_data[dev]['y'].append(y_data[i])

    for dev_id in sorted(plot_data.keys()):
        if plot_data[dev_id]['x']:
            ax.scatter(plot_data[dev_id]['x'], plot_data[dev_id]['y'], 
                        label=plot_data[dev_id]['label'], marker=plot_data[dev_id]['marker'], s=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_metric_name = metric_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "").replace("°", "")
    safe_xlabel = x_label.replace(" ", "_")
    savepath = os.path.join(savedir, f"{safe_metric_name}_vs_{safe_xlabel}_{ts}.png")
    
    try:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {savepath}")
    except Exception as e:
        print(f"Error saving plot: {e}", file=sys.stderr)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} <points_file> <metric_idx> <incr> <k> '<local_sizes>' '<local_sizes_ch>'", file=sys.stderr)
        print("NOTE: For <incr>=0, <local_sizes> should be a space-separated string (e.g., '16 32 64').", file=sys.stderr)
        sys.exit(1)

    points_file_path = sys.argv[1]
    metric_code = sys.argv[2]
    incr = int(sys.argv[3])
    k = int(sys.argv[4])
    # These are now strings that might contain space-separated values
    lss_str = sys.argv[5]
    lssc_str = sys.argv[6]

    if not metric_code.isdigit() or not (0 <= int(metric_code) <= 6):
        print(f"Error: Invalid metric index '{metric_code}'. Must be between 0 and 6.", file=sys.stderr)
        sys.exit(1)

    metric_name = METRIC_NAMES.get(metric_code, f"Metric_{metric_code}")
    parsed_data = parse_points_file(points_file_path)

    if not parsed_data or not parsed_data['metric']:
        print("Error: No valid data could be parsed from the points file.", file=sys.stderr)
        sys.exit(1)

    # --- Conditional Plotting Logic ---
    if incr == 0:
        # If increment is 0, parse the local size strings into lists of numbers
        try:
            lss_values = [int(x) for x in lss_str.split()]
            lssc_values = [int(x) for x in lssc_str.split()]
        except ValueError as e:
            print(f"Error: Could not parse local size arrays. Ensure they are space-separated numbers.", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            sys.exit(1)

        num_data_points = len(parsed_data['metric'])
        # --- CRITICAL --- Check if array sizes match the number of data points
        if len(lss_values) != num_data_points:
            print(f"Error: Mismatch between number of local_sizes ({len(lss_values)}) and data points in file ({num_data_points}).", file=sys.stderr)
            sys.exit(1)
        if len(lssc_values) != num_data_points:
            print(f"Error: Mismatch between number of local_sizes_ch ({len(lssc_values)}) and data points in file ({num_data_points}).", file=sys.stderr)
            sys.exit(1)
            
        n_for_title = parsed_data['N'][0] if parsed_data['N'] else 'Unknown'

        # Generate Plot 1: Metric vs. local_size
        title1 = f"{metric_name} vs. local_size, N={n_for_title}, k={k}"
        generate_plot(x_data=lss_values, y_data=parsed_data['metric'], dev_list=parsed_data['dev'],
                      x_label="local_size", y_label=metric_name, title=title1, metric_name=metric_name)
        
        # Generate Plot 2: Metric vs. local_size_ch
        title2 = f"{metric_name} vs. local_size_ch, N={n_for_title}, k={k}"
        generate_plot(x_data=lssc_values, y_data=parsed_data['metric'], dev_list=parsed_data['dev'],
                      x_label="local_size_ch", y_label=metric_name, title=title2, metric_name=metric_name)
    else:
        # Original behavior: plot Metric vs. N
        title_original = f"{metric_name} vs. N, Increment={incr}, k={k}, local_size={lss_str}, local_size_ch={lssc_str}"
        generate_plot(x_data=parsed_data['N'], y_data=parsed_data['metric'], dev_list=parsed_data['dev'],
                      x_label="N", y_label=metric_name, title=title_original, metric_name=metric_name)