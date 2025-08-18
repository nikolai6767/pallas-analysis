import matplotlib.pyplot as plt #type: ignore
import csv
import os
import glob


csv_folder = "../../run_benchmarks/run_nas_benchmark/details/details"
output_folder = "../plot/details_nas"


for csv_filename in glob.glob(os.path.join(csv_folder, "*.csv")):
    times = []
    sizes = []

    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 4:
                try:
                    time = int(row[1])
                    size=int(row[2])
                    sizes.append(size)
                    times.append(time)
                except ValueError:
                    continue 

    if sizes and times:
        plt.figure(figsize=(10, 5))
        plt.title("Temps en fonction de la taille")
        plt.ylabel("Taille (en octets)")
        plt.xlabel("Durée (en ns)")
        plt.scatter(times, sizes, s=10, c='red', alpha=0.6)
        plt.grid(True)
        # plt.xscale('log') 
        plt.yscale('log')

        plt.tight_layout()

        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        output_path = os.path.join(output_folder, f"{base_name}.png")
        plt.savefig(output_path)
        plt.close()