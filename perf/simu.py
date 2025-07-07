import pandas as pd
import matplotlib.pyplot as plt
import io

data = """
function,calls,total,min,max,average
PRINT_EVENT,12067968,1.79498e+10,950,1.68418e+06,1487.39
PRINT_EVENT_PRINT_TIMESTAMP,12067968,7.60938e+09,430,1.64775e+06,630.543
PRINT_EVENT_GET_NAME,12067968,2.2252e+09,120,194575,184.389
PRINT_EVENT_GET_TOKEN_STRING,12067968,-1.11025e+26,-9.22337e+18,0,-9.19998e+18
PRINT_EVENT_GET_EVENT_STRING,12067968,5.94738e+09,110,1.68338e+06,492.823
PRINT_EVENT_GET_PRINT_EV_ATT,12067968,1.0606e+09,80,145274,87.8854
PRINT_EV_ATT,12067968,3.11077e+08,20,87372,25.7771
PRINT_EVENT_ENDL,12067968,4.9126e+08,30,156484,40.7078
"""

df = pd.read_csv(io.StringIO(data))

plt.figure(figsize=(14, 6))
bar_width = 0.25
index = range(len(df))

plt.bar([i - bar_width for i in index], df["min"], width=bar_width, label="min (ns)", color='blue')
plt.bar(index, df["average"], width=bar_width, label="average (ns)", color='orange')
plt.bar([i + bar_width for i in index], df["max"], width=bar_width, label="max (ns)", color='red')

plt.xticks(index, df["function"], rotation=45, ha="right")
plt.ylabel("Temps (ns)")
plt.title("Stats temporelles par fonction")
plt.legend()
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.show()
