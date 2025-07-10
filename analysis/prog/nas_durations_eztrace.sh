#!/bin/bash

input="$PWD/../res/nas_durations.csv"
output="$PWD/../res/nas_eztrace_time_mean.csv"
echo -n > $output
# En-tête


echo "NAME,MEAN_EZTRACE" > "$output"



current_algo=""
index=0
sum=0
count=0

while IFS=',' read -r algo time; do
    if [ "$algo" != "$current_algo" ]; then
        if [ "$count" -gt 0 ]; then
            mean=$(echo "$sum / $count" | bc -l)
            echo "$current_algo,$mean" >> "$output"
        fi
        current_algo="$algo"
        index=1
        sum=0
        count=0
    else
        index=$((index + 1))
    fi

    if [ $((index % 2)) -eq 1 ]; then
        sum=$(echo "$sum + $time" | bc -l)
        count=$((count + 1))
    fi
done < <(tail -n +2 "$input")

if [ "$count" -gt 0 ]; then
    mean=$(echo "$sum / $count" | bc -l)
    echo "$current_algo,$mean" >> "$output"
fi

