import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier
df = pd.read_csv("res.csv")

# Optionnel : trier par nombre d'appels
df = df.sort_values(by="calls", ascending=False)
fonctions = df["function"]

fig, ax1 = plt.subplots(figsize=(14, 6))

# Axe 1 : Nombre d'appels
color = 'tab:blue'
ax1.set_xlabel('Fonction')
ax1.set_ylabel('Appels', color=color)
ax1.bar(fonctions, df["calls"], color=color, alpha=0.6, label="calls")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(range(len(fonctions)))
ax1.set_xticklabels(fonctions, rotation=45, ha='right')

# Axe 2 : Temps en ns
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Temps (ns)', color=color2)
ax2.plot(fonctions, df["min"], 'g--o', label="min")
ax2.plot(fonctions, df["average"], 'orange', marker='o', label="average")
ax2.plot(fonctions, df["max"], 'r--o', label="max")
ax2.tick_params(axis='y', labelcolor=color2)

# Légende
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.title("Statistiques d'appel de fonctions")
plt.tight_layout()
ax2.set_yscale("log")
plt.show()

