import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

size = pd.read_csv("../res/nas_trace_size_mean.csv")
overhead= pd.read_csv("../res/nas_overhead_mean.csv")
eztrace = pd.read_csv("../res/nas_eztrace_time_mean.csv")  
vanilla = pd.read_csv("../res/nas_vanilla_time_mean.csv")
size=pd.read_csv("../res/nas_trace_size_mean.csv")
size=size.sort_values(by="NAME")


df = pd.merge(vanilla, eztrace, on="NAME", suffixes=('_vanilla', '_eztrace'))
df = pd.merge(df, overhead, on="NAME")
df=pd.merge(df, size, on="NAME")

df["MAX_EZTRACE"]= df["MAX_EZTRACE"] / df["MEAN_VANILLA"]
df["MIN_EZTRACE"]= df["MIN_EZTRACE"] / df["MEAN_VANILLA"]
df["MEAN_EZTRACE"]= df["MEAN_EZTRACE"] / df["MEAN_VANILLA"]

df["MAX_VANILLA"]= df["MAX_VANILLA"] / df["MEAN_VANILLA"]
df["MIN_VANILLA"]= df["MIN_VANILLA"] / df["MEAN_VANILLA"]
df["MEAN_VANILLA"]= df["MEAN_VANILLA"] / df["MEAN_VANILLA"]

for i in range(len(df) - 1):
    df.at[i, "NAME"] = "NAS " + df.at[i, "NAME"].upper()
df.at[len(df) - 1, "NAME"] = df.at[len(df) - 1, "NAME"].upper()

df["OVH_PERCENT"] = ((df["MEAN_EZTRACE"] - df["MEAN_VANILLA"]) / df["MEAN_VANILLA"]) * 100

ind = np.arange(len(df))
width = 0.35

fig, ax = plt.subplots(figsize=(12,6))

bars_vanilla = ax.bar(ind - width/2, df["MEAN_VANILLA"], width, label='Vanilla', color='tab:blue')

bars_eztrace = ax.bar(ind + width/2, df["MEAN_EZTRACE"], width, label='Eztrace', color='tab:orange')


yerr_lower = df["MEAN_EZTRACE"] - df["MIN_EZTRACE"]
yerr_upper = df["MAX_EZTRACE"] - df["MEAN_EZTRACE"]
ax.errorbar(ind + width/2, df["MEAN_EZTRACE"], 
            yerr=[yerr_lower, yerr_upper],label="_nolegend_",
            fmt='none', ecolor='black', capsize=5)

yerr_low = df["MEAN_VANILLA"] - df["MIN_VANILLA"]
yerr_up = df["MAX_VANILLA"] - df["MEAN_VANILLA"]
ax.errorbar(ind - width/2, df["MEAN_VANILLA"], 
            yerr=[yerr_low, yerr_up],
            fmt='none', ecolor='black', capsize=5, label="_nolegend_")


for i, row in df.iterrows():
    y_text = row["MAX_EZTRACE"] + 0.03 * max(df["MAX_EZTRACE"]) if 'MAX_EZTRACE' in df.columns else row["MEAN_EZTRACE"] + 0.05 * max(df["MEAN_EZTRACE"])
    ax.text(ind[i] + width/2, y_text,
        f'+{row["OVH_PERCENT"]:.1f}%', ha='center', va='bottom', fontsize=8, color='red')


ax.set_ylabel('Temps (s)')
ax.set_title('Comparaison en temps Vanilla vs Eztrace avec Overhead sur 20 simulations')
ax.set_xticks(ind)
ax.set_xticklabels(df["NAME"], rotation=30, ha='center')
ax.legend()

ax.legend(loc='upper left', ncol=2, frameon=True)
plt.tight_layout()
plt.savefig("../plot/eztrace_overhead.pdf")
plt.show()