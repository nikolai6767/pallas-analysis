import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../res/nas_comp_write.csv")
colonnes = ['n_calls', 'total_time', 'min', 'max', 'mean']


for col in colonnes:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '.', regex=False)  
        .str.replace(r'[^\d\.]', '', regex=True) 
        .astype(float)             
    )

mean = df.groupby(['source', 'algo'])[colonnes].mean()

mean.to_csv("../res/nas_comp_write_mean.csv", index=False)


df['err_bars_inf'] = df['mean'] - df['min']
df['err_bars_sup'] = df['max'] - df['mean']

plt.figure(figsize=(14, 7))



plt.title("Temps moyen par fonction et groupe (avec min/max)")
plt.ylabel("Temps meanen")
plt.xlabel("Fonction")
plt.xticks(rotation=45)
plt.legend(title="Groupe")
plt.tight_layout()
plt.savefig("barplot_meanenne_min_max_par_groupe.pdf")
plt.show() 