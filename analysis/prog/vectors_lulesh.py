import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm



summary = []

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
        func = df.iloc[:, 0].dropna().astype(str).mode()[0]
        alg = sf.split("-")[-1]
        summary.append({"func": func, "alg": alg, "mean_duration_ns": times.mean()})



    summary_df = pd.DataFrame(summary)

    agg = summary_df.groupby(["func", "alg"], observed=True).agg(mean_duration_ns=("mean_duration_ns", "mean"), samples=("mean_duration_ns", "size")).reset_index()

    order = [sf.split("-")[-1] for sf in subfolders]
    agg["alg"] = pd.Categorical(agg["alg"], categories=order, ordered=True)

    output_dir = os.path.join("../plot/", "lulesh_vectors_summary")
    os.makedirs(output_dir, exist_ok=True)

    for func_name, grp in agg.groupby("func"):
        grp = grp.sort_values("alg")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(grp["alg"].astype(str), grp["mean_duration_ns"])
        ax.set_yscale("log")
        ax.set_xlabel("Subvector Size")
        ax.set_ylabel("Mean duration (ns)")
        ax.set_title(f"Time for '{func_name}'")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{func_name}.png"), dpi=300)
        plt.close()




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
