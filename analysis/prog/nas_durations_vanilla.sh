#!/bin/bash

input="$PWD/../res/nas_durations.csv"
output="$PWD/../res/nas_vanilla_time_mean.csv"
echo -n > $output
# En-tête


echo "NAME,MEAN_VANILLA,MIN_VANILLA,MAX_VANILLA" > "$output"



current_algo=""
index=0
sum=0
count=0
min_time=0
max_time=0

while IFS=',' read -r algo time; do
    if [ "$algo" != "$current_algo" ]; then
        if [ "$count" -gt 0 ]; then
            mean=$(echo "$sum / $count" | bc -l)
            echo "$current_algo,$mean,$min_time,$max_time" >> "$output"
        fi
        current_algo="$algo"
        index=1
        sum=0
        count=0
        min_time=$time
        max_time=$time
    else
        index=$((index + 1))
    fi

    if [ $((index % 2)) -eq 0 ]; then
        sum=$(echo "$sum + $time" | bc -l)
        count=$((count + 1))

        min_time=$(echo "$min_time $time" | awk '{if ($2 < $1) print $2; else print $1}')
        max_time=$(echo "$max_time $time" | awk '{if ($2 > $1) print $2; else print $1}')
    fi
done < <(tail -n +2 "$input")

if [ "$count" -gt 0 ]; then
    mean=$(echo "$sum / $count" | bc -l)
    echo "$current_algo,$mean,$min_time,$max_time" >> "$output"
fi