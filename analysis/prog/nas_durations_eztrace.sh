#!/bin/bash

input="$PWD/../res/durations.csv"
output="$PWD/../res/nas_eztrace_time_mean.csv"
echo -n > $output


echo "NAME,MEAN_EZTRACE,MIN_EZTRACE,MAX_EZTRACE" > "$output"


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
        sum=$time
        count=1
        min_time=$time
        max_time=$time
    else
        index=$((index + 1))
    fi

    if [ $((index % 2)) -eq 1 ]; then
        sum=$(echo "$sum + $time" | bc -l)
        count=$((count + 1))

        min_time=$(echo "$min_time $time" | awk '{if ($2 < $1) print $2; else print $1}')
        max_time=$(echo "$max_time $time" | awk '{if ($2 > $1) print $2; else print $1}')
    fi

done < <(tail -n +2 "$input")


if [ "$count" -gt 1 ]; then
    mean=$(echo "$sum / $count" | bc -l)
    echo "$current_algo,$mean,$min_time,$max_time" >> "$output"
fi

