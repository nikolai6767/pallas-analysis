# import matplotlib.pyplot as plt  # type: ignore
# import csv
# import os
# import glob
# from collections import defaultdict
# from statistics import mean

# # Dossiers
# csv_folder = "../../run_benchmarks/run_nas_benchmark/20_iter/details"
# output_folder = "../plot/"
# os.makedirs(output_folder, exist_ok=True)


# func_times = defaultdict(list)          
# func_counts_per_file = defaultdict(list) 

# for csv_filename in sorted(glob.glob(os.path.join(csv_folder, "*.csv"))):
#     local_counts = defaultdict(int) 

#     with open(csv_filename, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             if len(row) >= 2:
#                 func = row[0].strip()
#                 try:
#                     t = int(row[1])
#                 except ValueError:
#                     continue
#                 func_times[func].append(t)
#                 local_counts[func] += 1

#     for func, cnt in local_counts.items():
#         func_counts_per_file[func].append(cnt)

# stats = []
# for func, times in func_times.items():
#     if not times:
#         continue
#     counts_list = func_counts_per_file.get(func, [])
#     if counts_list:
#         n_avg_per_file = mean(counts_list)
#     else:
#         n_avg_per_file = 0.0
#     stats.append((func, mean(times), n_avg_per_file))


# labels      = [s[0] for s in stats]
# mean_values = [s[1] for s in stats]
# n_avg_list  = [s[2] for s in stats]


# plt.figure(figsize=(12, 6))
# ax = plt.gca()

# x = range(len(labels))
# bars = ax.bar(x, mean_values, label="Durée moyenne (ns)")

# ax.set_yscale("log")
# ax.set_ylabel("Durée (ns)")
# ax.set_title("Durée par fonction")
# ax.set_xticks(list(x))
# ax.set_xticklabels(labels, rotation=60, ha="right")
# ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)
# ax.legend()

# ymin, ymax = ax.get_ylim()
# ax.set_ylim(ymin, ymax * 1.25)

# for rect, n_avg in zip(bars, n_avg_list):
#     h = rect.get_height()
#     n_int = int(round(n_avg)) 
#     ax.text(rect.get_x() + rect.get_width()/2,
#             h * 1.05 if h > 0 else 1,
#             f"n={n_int}",
#             ha="center", va="bottom", fontsize=8)

# plt.tight_layout()
# out_png = os.path.join(output_folder, "nas_pcw_details_bis.png")
# plt.savefig(out_png, dpi=300)
# plt.close()






















import matplotlib.pyplot as plt  # type: ignore
import csv
import os
import glob
from collections import defaultdict
from statistics import mean
import numpy as np

details_dir = "../../run_benchmarks/run_nas_benchmark/20_iter/details"
output_folder = "../plot/"
os.makedirs(output_folder, exist_ok=True)

func_times = defaultdict(lambda: defaultdict(list))
func_counts_per_file = defaultdict(lambda: defaultdict(list))

csv_files = sorted(glob.glob(os.path.join(details_dir, "*.C.*.csv")))
if not csv_files:
    raise RuntimeError(f"Aucun fichier CSV trouvé avec le motif : {os.path.join(details_dir, '*.C.*.csv')}")

for path in csv_files:
    base = os.path.basename(path)
    name_wo_ext, _ = os.path.splitext(base)
    prefix = name_wo_ext.split(".C.", 1)[0]
    if "_" not in prefix:
        continue
    func, algo = prefix.rsplit("_", 1)

    local_count = 0
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                try:
                    t = int(row[1])
                except ValueError:
                    continue
                func_times[algo][func].append(t)
                local_count += 1
    if local_count > 0:
        func_counts_per_file[algo][func].append(local_count)

algos = sorted(func_times.keys())
all_funcs = sorted({f for a in algos for f in func_times[a].keys()})

mean_by_algo_func = {a: {} for a in algos}
navg_by_algo_func = {a: {} for a in algos}

for a in algos:
    for f in all_funcs:
        times = func_times[a].get(f, [])
        mean_by_algo_func[a][f] = mean(times) if times else np.nan
        counts = func_counts_per_file[a].get(f, [])
        navg_by_algo_func[a][f] = mean(counts) if counts else np.nan

plt.figure(figsize=(14, 7))
ax = plt.gca()

n_funcs = len(all_funcs)
n_algos = len(algos)

indices = np.arange(n_funcs)
group_width = 0.85
bar_width = group_width / max(n_algos, 1)

bars_per_algo = []

for i, a in enumerate(algos):
    heights = [mean_by_algo_func[a][f] for f in all_funcs]
    x_pos = indices - (group_width / 2) + i * bar_width + bar_width / 2
    bars = ax.bar(x_pos, heights, width=bar_width, label=a)
    bars_per_algo.append((a, bars))

ax.set_yscale("log")
ax.set_ylabel("Durée (ns)")
ax.set_title("Durée par fonction — comparaison par algorithme")
ax.set_xticks(indices)
ax.set_xticklabels(all_funcs, rotation=60, ha="right")
ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)
ax.legend(title="Algorithmes", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, ncol=1)

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.25)

for i, a in enumerate(algos):
    _, bars = bars_per_algo[i]
    for rect, f in zip(bars, all_funcs):
        h = rect.get_height()
        n_avg = navg_by_algo_func[a][f]
        if np.isfinite(h) and h > 0 and np.isfinite(n_avg):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h * 1.05,
                f"{int(round(n_avg))}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

plt.tight_layout()
out_png = os.path.join(output_folder, "nas_pcw_details.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()