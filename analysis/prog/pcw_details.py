import matplotlib.pyplot as plt  # type: ignore
import csv
import os
import glob
from collections import defaultdict
from statistics import mean

# Dossiers
csv_folder = "../../run_benchmarks/run_nas_benchmark/pcw/details"
output_folder = "../plot/"
os.makedirs(output_folder, exist_ok=True)


func_times = defaultdict(list)          
func_counts_per_file = defaultdict(list) 

for csv_filename in sorted(glob.glob(os.path.join(csv_folder, "*.csv"))):
    local_counts = defaultdict(int) 

    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                func = row[0].strip()
                try:
                    t = int(row[1])
                except ValueError:
                    continue
                func_times[func].append(t)
                local_counts[func] += 1

    for func, cnt in local_counts.items():
        func_counts_per_file[func].append(cnt)

stats = []
for func, times in func_times.items():
    if not times:
        continue
    counts_list = func_counts_per_file.get(func, [])
    if counts_list:
        n_avg_per_file = mean(counts_list)
    else:
        n_avg_per_file = 0.0
    stats.append((func, mean(times), n_avg_per_file))


labels      = [s[0] for s in stats]
mean_values = [s[1] for s in stats]
n_avg_list  = [s[2] for s in stats]


plt.figure(figsize=(12, 6))
ax = plt.gca()

x = range(len(labels))
bars = ax.bar(x, mean_values, label="Durée moyenne (ns)")

ax.set_yscale("log")
ax.set_ylabel("Durée (ns)")
ax.set_title("Durée par fonction")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=60, ha="right")
ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)
ax.legend()

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.25)

for rect, n_avg in zip(bars, n_avg_list):
    h = rect.get_height()
    n_int = int(round(n_avg)) 
    ax.text(rect.get_x() + rect.get_width()/2,
            h * 1.05 if h > 0 else 1,
            f"n={n_int}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_png = os.path.join(output_folder, "nas_pcw_details.png")
plt.savefig(out_png, dpi=300)
plt.close()
