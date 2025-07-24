import matplotlib.pyplot as plt
import csv

# csv_filename = "../../run_benchmarks/run_nas_benchmark/details/write_details_bt.C.x.csv"

# times = []
# sizes = []


# with open(csv_filename, newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         if len(row) >= 4:
#             try:
#                 size = int(row[1])
#                 time = int(row[2])*int(row[3])
#                 sizes.append(size)
#                 times.append(time)
#             except ValueError:
#                 pass

# plt.figure(figsize=(10, 5))
# plt.title("Temps en fonction de la taille")
# plt.xlabel("Taille (en octets)")
# plt.ylabel("Durée (en ns)")
# plt.scatter(sizes, times, s=10, c='red', alpha=0.6)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("../plot/nas_details.pdf")



import os
import glob


csv_folder = "../../run_benchmarks/run_nas_benchmark/details/"
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
                    if csv_filename.startswith("write_details_"):
                        size = int(row[2])*int(row[3])
                    elif csv_filename.startswith("zstd"):
                        size = int(row[2])*int(row[3])
                    elif csv_filename.startswith("write_duration_vector"):
                        size = int(row[2])*int(row[3])
                    else:
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
        output_path = os.path.join(output_folder, f"{base_name}.pdf")
        plt.savefig(output_path)
        plt.close()