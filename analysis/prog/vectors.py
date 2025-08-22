import os
import re
import numpy as np # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import csv
from collections import defaultdict

summary = []

base_dir = os.path.expanduser("~/soft/pallas-analysis/run_benchmarks/run_nas_benchmark/vectors")
subfolders = sorted([f for f in os.listdir(base_dir) if f.startswith("000")])
perf_files = []
n_subfolders = len(subfolders)
cmap_size = plt.get_cmap('tab10', n_subfolders)
order = [sf.split("-")[-1] for sf in subfolders] 

perf_info = {}
for sf in subfolders:
    size_label = sf.split("-")[-1]
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
    perf_info[size_label] = {"TIME_s": mean_time_s, "MAX_PERF_kB": mean_max_perf_kB}

first_details_path = os.path.join(base_dir, subfolders[0], "details")
file_names = [f for f in os.listdir(first_details_path) if f.endswith(".csv")]

fname_re = re.compile(r'^(?P<func>.+)_(?P<alg>[a-z]+)\.C\.x\.csv$')

summaries_by_func = defaultdict(list)

for file_name in file_names:
    m = fname_re.match(file_name)
    if not m:
        continue
    func_name = m.group('func')
    alg_name = m.group('alg')

    summary = []
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False

    for i, sf in enumerate(subfolders):
        size_label = sf.split("-")[-1]
        path = os.path.join(base_dir, sf, "details", file_name)
        if not os.path.isfile(path):
            continue

        df = pd.read_csv(path, header=None, sep=None, engine="python", on_bad_lines="skip")
        df = df.iloc[:, :3]

        def _to_num(series):

            s = series.astype(str).str.replace(r"[^\d.+\-eE]", "", regex=True)
            return pd.to_numeric(s, errors="coerce")

        df[1] = _to_num(df.iloc[:, 1])
        df[2] = _to_num(df.iloc[:, 2])

        df = df.dropna(subset=[1, 2])
        df = df[(df[1] > 0) & (df[2] > 0)]
        if df.empty:
            continue


        med_time, med_size = df[1].median(), df[2].median()
        if (med_size < 32) and (med_time > 1e5):
            df[1], df[2] = df[2].copy(), df[1].copy()

        if len(df) >= 20:
            lo_t, hi_t = df[1].quantile([0.02, 0.98])
            lo_s, hi_s = df[2].quantile([0.02, 0.98])
            df = df[(df[1] >= lo_t) & (df[1] <= hi_t) & (df[2] >= lo_s) & (df[2] <= hi_s)]
            if df.empty:
                continue

        times = df[1].astype(float)
        sizes = df[2].astype(float)


        color = cmap_size(i)
        ax.scatter(times, sizes, s=20, alpha=0.4,
                   label=size_label, rasterized=True, zorder=2, color=color)
        ax.scatter(times.mean(), sizes.mean(), s=100, marker='o',
                   edgecolors='black', color=color, zorder=10)
        n = times.notna().sum()
        any_data = True

        perf = perf_info.get(size_label, {})
        summary.append({
            "func": func_name,
            "alg": alg_name,
            "size": size_label,
            "mean_duration_ns": times.mean(),
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
    out_dir_overall = os.path.join("../plot/", "nas_vectors_by_func")
    os.makedirs(out_dir_overall, exist_ok=True)
    plt.savefig(os.path.join(out_dir_overall, file_name.replace(".csv", ".png")), dpi=300)
    plt.close()

    summaries_by_func[func_name].extend(summary)

out_dir_summary = os.path.join("../plot/", "nas_vectors_summary_by_func")
os.makedirs(out_dir_summary, exist_ok=True)

for func_name, rows in summaries_by_func.items():
    if not rows:
        continue
    df = pd.DataFrame(rows)

    agg = df.groupby(["func", "size", "alg"], observed=True).agg(
        mean_duration_ns=("mean_duration_ns", "mean"),
        median_duration_ns=("median_duration_ns", "median"),
        mean_TIME_s=("mean_TIME_s", "mean"),
        mean_MAX_PERF_kB=("mean_MAX_PERF_kB", "mean"),
        n=("n", "sum"),
    ).reset_index()

    size_order = [s for s in order if s in set(agg["size"])]
    if not size_order:
        continue
    agg["size"] = pd.Categorical(agg["size"], categories=size_order, ordered=True)
    agg = agg.sort_values(["size", "alg"])

    algs_for_func = []
    for fn in file_names:
        m = fname_re.match(fn)
        if m and m.group('func') == func_name:
            a = m.group('alg')
            if a not in algs_for_func:
                algs_for_func.append(a)
    if not algs_for_func:
        algs_for_func = sorted(agg["alg"].unique().tolist())

    x = np.arange(len(size_order))
    total_width = 0.8
    n_alg = max(1, len(algs_for_func))
    bar_width = total_width / n_alg
    offsets = (np.arange(n_alg) - (n_alg - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap_alg = plt.get_cmap('tab10', n_alg)
    handles = []

    for ai, alg in enumerate(algs_for_func):
        heights = []
        medians = []
        for sz in size_order:
            sel = agg[(agg["size"] == sz) & (agg["alg"] == alg)]
            if not sel.empty:
                heights.append(sel["mean_duration_ns"].iloc[0])
                medians.append(sel["median_duration_ns"].iloc[0])
            else:
                heights.append(np.nan)
                medians.append(np.nan)
        bars = ax.bar(x + offsets[ai], heights, width=bar_width, label=alg, color=cmap_alg(ai), zorder=3)
        handles.append(bars)

        ax.scatter(x + offsets[ai], medians, marker="x", s=40, zorder=5)


    ax.set_yscale("log")
    ax.set_ylabel("Duration (ns)")
    ax.set_xlabel("Vector size")
    ax.set_title(f"{func_name} — grouped by vector size")
    ax.set_xticks(x)
    ax.set_xticklabels(size_order, rotation=0)
    ax.grid(True, axis="y", zorder=0)

    ax2 = ax.twinx()
    time_ns = [perf_info.get(sz, {}).get("TIME_s", np.nan) for sz in size_order]
    ax2.set_yscale("log")
    ax2.scatter(x, time_ns, marker="x", s=60, zorder=6, color="black", label="App total duration (s)")
    ax2.set_ylabel("App total duration (s)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize="small", ncol=min(4, len(lines1)+len(lines2)), loc="upper center", bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_summary, f"{func_name}.png"), dpi=300)
    plt.close()
