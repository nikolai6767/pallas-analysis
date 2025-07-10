import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

size = pd.read_csv("../res/nas_trace_size_mean.csv")
overhead= pd.read_csv("../res/nas_overhead_mean.csv")
eztrace = pd.read_csv("../res/nas_eztrace_time_mean.csv")  
vanilla = pd.read_csv("../res/nas_vanilla_time_mean.csv")


df = pd.merge(vanilla, eztrace, on="NAME", suffixes=('_vanilla', '_eztrace'))
df = pd.merge(df, overhead, on="NAME")

df["OVH_PERCENT"] = (df["MEAN_OVH"] / df["MEAN_VANILLA"]) * 100

ind = np.arange(len(df))
width = 0.35

fig, ax = plt.subplots(figsize=(12,6))

bars_vanilla = ax.bar(ind - width/2, df["MEAN_VANILLA"], width, label='Vanilla', color='tab:blue')

bars_eztrace = ax.bar(ind + width/2, df["MEAN_EZTRACE"], width, label='Eztrace', color='tab:orange')

for i, row in df.iterrows():
    ax.text(ind[i] + width/2, row["MEAN_EZTRACE"] + 0.05 * max(df["MEAN_EZTRACE"]),
            f'+{row["OVH_PERCENT"]:.1f}%', ha='center', va='bottom', fontsize=8, color='red')

ax.set_xlabel('Algorithme')
ax.set_ylabel('Temps (s)')
ax.set_title('Comparaison Vanilla vs Eztrace avec Overhead')
ax.set_xticks(ind)
ax.set_xticklabels(df["NAME"], rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig("../plot/eztrace_overhead.pdf")
plt.show()