import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../res/nas_comp_write.csv")
colonnes = ['n_calls', 'total_time', 'min', 'max', 'mean']


mean = df.groupby(['source', 'algo'])[colonnes].mean().reset_index()

mean.to_csv("../res/nas_comp_write_mean.csv", index=False)


err_inf = mean['mean'] - mean['min']
err_sup = mean['max'] - mean['mean']


mean['label'] = mean['algo'] + " | " + mean['source']

plt.figure(figsize=(14, 7))
plt.bar(
    x=mean['label'],
    height=mean['mean'],
    yerr=[err_inf, err_sup],
    capsize=5,
    color='skyblue',
    edgecolor='black'
)

plt.title("Temps moyen par (algo, source) avec min/max", fontsize=14)
plt.ylabel("Temps moyen", fontsize=12)
plt.xlabel("Algo | Source", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.yscale('log')
plt.savefig("../plot/nas_comp_write.pdf")
plt.show()
