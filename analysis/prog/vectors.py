import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm





base_dir = os.path.expanduser("~/soft/pallas-analysis/run_benchmarks/run_nas_benchmark/vectors")
subfolders = sorted([f for f in os.listdir(base_dir) if f.startswith("000")])

n_subfolders = len(subfolders)
cmap = cm.get_cmap('tab10', n_subfolders)

first_details_path = os.path.join(base_dir, subfolders[0], "details")
file_names = [f for f in os.listdir(first_details_path) if f.endswith(".csv")]

for file_name in file_names:
    plt.figure(figsize=(10, 6))
    all_rows = []

    for i, subfolder in enumerate(subfolders):
        details_path = os.path.join(base_dir, subfolder, "details")
        file_path = os.path.join(details_path, file_name)
        color=cmap(i)
        times = []
        sizes = []

        try:
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)

                for row in reader:
                    if len(row) >= 4:
                        try:
                            time = int(row[1])
                            if file_name.startswith(("write_details_", "zstd", "write_duration_vector")):
                                size = int(row[2]) * int(row[3])
                            else:
                                size = int(row[2])

                            times.append(time)
                            sizes.append(size)
                            all_rows.append(row)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Erreur dans {file_path} : {e}")
            continue

        if times and sizes:
            plt.scatter(times, sizes, s=40, alpha=0.2, label=subfolder, rasterized=True, color=color)

            df = pd.DataFrame(all_rows, columns=header)
            df = df.apply(pd.to_numeric, errors='coerce')

            try:
                mean_time = df.iloc[:, 1].mean()
                if file_name.startswith(("write_details_", "zstd", "write_duration_vector")):
                    mean_size = (df.iloc[:, 2] * df.iloc[:, 3]).mean()
                else:
                    mean_size = df.iloc[:, 2].mean()
                label = subfolder.split("-")[-1]

                if pd.notna(mean_time) and pd.notna(mean_size):

                    plt.scatter(mean_time, mean_size, s=100, color='black', marker='x')

                    plt.annotate(
                        label,
                        (mean_time, mean_size),
                        textcoords="offset points",
                        xytext=offset,
                        ha='center',
                        fontsize=8,
                        color="color"
                    )
           
            except Exception as e:
                continue

    plt.title(f"{file_name} — Temps en fonction de la taille")
    plt.xlabel("Durée (en ns)")
    plt.ylabel("Taille (en octets)")
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title="Size", fontsize='small', loc='best')
    plt.tight_layout()

    output_dir = os.path.join(base_dir, "plots_by_file")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name.replace(".csv", ".png"))
    if plt.gca().has_data():
        plt.savefig(output_file, dpi=300)
    plt.close()
