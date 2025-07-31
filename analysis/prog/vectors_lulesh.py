import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm





base_dir = os.path.expanduser("~/soft/pallas-analysis/run_benchmarks/run_lulesh/vectors")
subfolders = sorted([f for f in os.listdir(base_dir) if f.startswith("000")])

n_subfolders = len(subfolders)
cmap = cm.get_cmap('tab10', n_subfolders)

first_details_path = os.path.join(base_dir, subfolders[0], "details")
file_names = [f for f in os.listdir(first_details_path) if f.endswith(".csv")]


for file_name in file_names:
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False

    for i, sf in enumerate(subfolders):
        path = os.path.join(base_dir, sf, "details", file_name)
        if not os.path.isfile(path):
            continue

        df = pd.read_csv(path, header=None)
        df = df.dropna(subset=[1, 2])
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=[1, 2])

        if df.empty:
            continue

        if file_name.startswith(("write_details_", "write_duration_vector")) and df.shape[1] >= 4:
            sizes = df.iloc[:, 2] * df.iloc[:, 3]
        elif file_name.startswith("zstd"):
            sizes = df.iloc[:, 2] * df.iloc[:, 4]
        else:
            sizes = df.iloc[:, 2]
        times = df.iloc[:, 1]

        color = cmap(i)
        ax.scatter(times, sizes, s=20, alpha=0.4,
                   label=sf.split("-")[-1], rasterized=True, zorder=2, color=color)
        ax.scatter(times.mean(), sizes.mean(), s=100, marker='o',
                   edgecolors='black', color=color, zorder=10)

        any_data = True


    plt.title(f"{file_name} — Size vs Duration")
    plt.xlabel("Duration (ns)")
    plt.ylabel("Size (byte)")
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title="Subvector size", fontsize='small', loc='best')
    plt.tight_layout()

    output_dir = os.path.join("../plot/", "lulesh_vectors")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name.replace(".csv", ".png"))
    plt.savefig(output_file, dpi=300) ### TODO: check dpi ###
    plt.close()
