import os
import re
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import csv

summary = []

base_dir = os.path.expanduser("~/soft/pallas-analysis/run_benchmarks/run_nas_benchmark/vectors")
subfolders = sorted([f for f in os.listdir(base_dir) if f.startswith("000")])
perf_files = []
n_subfolders = len(subfolders)
cmap = plt.get_cmap('tab10', n_subfolders)
order = [sf.split("-")[-1] for sf in subfolders]


perf_info = {}
for sf in subfolders:
    alg = sf.split("-")[-1]
    perf_path = os.path.join(base_dir, sf, "perfo.csv")
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
    summary = []
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

        sizes = df.iloc[:, 2]
        times = df.iloc[:, 1]

        color = cmap(i)
        ax.scatter(times, sizes, s=20, alpha=0.4,
                   label=sf.split("-")[-1], rasterized=True, zorder=2, color=color)
        ax.scatter(times.mean(), sizes.mean(), s=100, marker='o',
                   edgecolors='black', color=color, zorder=10)
        n = times.notna().sum()
        any_data = True
        func = df.iloc[:, 0].dropna().astype(str).mode()[0]
        alg = sf.split("-")[-1]

        perf = perf_info.get(alg, {})
        summary.append({"func": func, "alg": alg, "mean_duration_ns": times.mean(), 
            "mean_TIME_s": perf.get("TIME_s", pd.NA),
            "mean_MAX_PERF_kB": perf.get("MAX_PERF_kB", pd.NA),
            "median_duration_ns": times.median(),
            "n": int(n),
        })


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
        median_duration_ns=("median_duration_ns", "median"),
        mean_TIME_s=("mean_TIME_s", "mean"),
        mean_MAX_PERF_kB=("mean_MAX_PERF_kB", "mean"),
    ).reset_index()
    agg["alg"] = pd.Categorical(agg["alg"], categories=order, ordered=True)
    agg = agg.sort_values("alg")

    out_dir_summary = os.path.join("../plot/", "nas_vectors_summary")
    os.makedirs(out_dir_summary, exist_ok=True)
    for func_name, grp in agg.groupby("func"):
        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(grp["alg"].astype(str), grp["mean_duration_ns"], label="Mean duration (ns)", color="slateblue")
        ax.set_yscale("log")
        ax.set_ylabel("Mean duration (ns)")

        ax2 = ax.twinx()
        time_ns = grp["mean_TIME_s"]
        ax2.set_yscale("log")
        ax2.scatter(
            grp["alg"].astype(str),
            time_ns,
            marker="x",
            color="black",
            s=60,
            zorder=5,
            label="App total duration (s)",
        )
        ax2.set_ylabel("App total duration (s)")


        x_positions = range(len(grp))
        medians = grp["median_duration_ns"]
        ax.scatter(
            x_positions,
            medians,
            marker="x",
            s=40,
            label="Median duration (ns) ",
            zorder=6,
            linewidths=1.5,
            color="orange"
        )

        for rect, max_perf in zip(bars, grp["mean_MAX_PERF_kB"]):
            height = rect.get_height()
            if pd.notna(max_perf):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height * 1.005,
                    f"{max_perf:.1f} KB",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="red",
                )

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize="small")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_summary, f"{func_name}.png"), dpi=300)
        plt.close()
