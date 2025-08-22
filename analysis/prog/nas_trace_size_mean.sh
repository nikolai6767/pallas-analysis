#!/bin/sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark/20_iter
file=$PWD/../res/trace_size.csv


file="$PWD/../res/trace_size.csv"        
output="$PWD/../res/nas_trace_size_mean.csv" 
echo -n > $output
echo "NAME,MEAN_SIZE" > "$output"



tail -n +2 "$file" | while IFS=',' read -r size name; do
    algo=$(echo "$name" | cut -d '.' -f1)
    echo "$algo $size"
done | \
awk '{ sum[$1] += $2; count[$1]++ } END { for (a in sum) printf "%s,%d\n", a, sum[a]/count[a] + 0.5 }' | sort >> $output