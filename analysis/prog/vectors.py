import os
import re
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import csv
import matplotlib.cm as cm # type: ignore



summary = []

base_dir = os.path.expanduser("~/soft/pallas-analysis/run_benchmarks/run_nas_benchmark/vectors")
subfolders = sorted([f for f in os.listdir(base_dir) if f.startswith("000")])
perf_files = []
n_subfolders = len(subfolders)
cmap = cm.get_cmap('tab10', n_subfolders)
order = [sf.split("-")[-1] for sf in subfolders]


perf_info = {}
for sf in subfolders:
    alg = sf.split("-")[-1]
    perf_path = os.path.join(base_dir, sf, "perf.csv")
    if not os.path.isfile(perf_path):
        continue
    try:
        perf_df = pd.read_csv(perf_path, header=None)
    except Exception:
        continue
    if perf_df.shape[1] < 2:
        continue
    mean_time_s = pd.to_numeric(perf_df.iloc[:, 0], errors="coerce").dropna().mean()
    mean_max_perf_kB = pd.to_numeric(perf_df.iloc[:, 1], errors="coerce").dropna().mean()
    perf_info[alg] = {"TIME_s": mean_time_s, "MAX_PERF_kB": mean_max_perf_kB}


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

        perf = perf_info.get(alg, {})
        summary.append({"func": func, "alg": alg, "mean_duration_ns": times.mean(), 
            "mean_TIME_s": perf.get("TIME_s", pd.NA),
            "mean_MAX_PERF_kB": perf.get("MAX_PERF_kB", pd.NA),
        })




    if any_data:
        ax.set_title(f"{file_name} — Size vs Duration")
        ax.set_xlabel("Duration (ns)")
        ax.set_ylabel("Size (byte)")
        ax.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="Subvector size", fontsize="small", loc="best")
        plt.tight_layout()
        out_dir_overall = os.path.join("../plot/", "nas_vectors")
        os.makedirs(out_dir_overall, exist_ok=True)
        plt.savefig(os.path.join(out_dir_overall, file_name.replace(".csv", ".png")), dpi=300)
        plt.close()

    summary_df = pd.DataFrame(summary)

    agg = summary_df.groupby(["func", "alg"], observed=True).agg(
        mean_duration_ns=("mean_duration_ns", "mean"),
        mean_TIME_s=("mean_TIME_s", "mean"),
        mean_MAX_PERF_kB=("mean_MAX_PERF_kB", "mean"),
    ).reset_index()
    agg["alg"] = pd.Categorical(agg["alg"], categories=order, ordered=True)

    out_dir_summary = os.path.join("../plot/", "nas_vectors_summary")
    os.makedirs(out_dir_summary, exist_ok=True)
    for func_name, grp in agg.groupby("func"):
        grp = grp.sort_values("alg")
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(grp["alg"].astype(str), grp["mean_duration_ns"])
        ax.set_yscale("log")
        ax.set_xlabel("Subvector Size")
        ax.set_ylabel("Mean duration (ns)")
        ax.set_title(f"Time for '{func_name}'")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        plt.xticks(rotation=45, ha="right")

        time_ns = grp["mean_TIME_s"] * 1e9  # s en ns !!!!
        ax.scatter(
            grp["alg"].astype(str),
            time_ns,
            marker="x",
            edgecolors="red",
            facecolors="none",
            s=40,
            zorder=5,
            label="Total time (ns)",
        )

        for rect, max_perf in zip(bars, grp["mean_MAX_PERF_kB"]):
            height = rect.get_height()
            if pd.notna(max_perf):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height * 1.05,
                    f"{max_perf:.1f} KB",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.legend(fontsize="small")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_summary, f"{func_name}.png"), dpi=300)
        plt.close()





    plt.title(f"{file_name} — Size vs Duration")
    plt.xlabel("Duration (ns)")
    plt.ylabel("Size (byte)")
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title="Subvector size", fontsize='small', loc='best')
    plt.tight_layout()
    output_dir = os.path.join("../plot/", "nas_vectors")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, file_name.replace(".csv", ".png"))
    plt.savefig(output_file, dpi=300) ### TODO: check dpi ###
    plt.close()
