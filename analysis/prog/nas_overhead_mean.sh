#!/bin/sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark/20_iter
file=$PWD/../res/overhead.csv

exit=$PWD/../res/nas_overhead_mean.csv
echo -n > "$exit"

ref_name=""
count=0
sum=0

echo "NAME,MEAN_OVH" >> $exit
tail -n +2 "$file" > /tmp/nas_tmp_input.csv
while IFS= read -r line ; do

    name=$(echo $line | cut -d ',' -f1)
    ovh=$(echo $line | cut -d ',' -f2)

    group=$(echo "$name" | cut -d '.' -f1)

    if [ -z "$ref_name" ]; then
        ref_name=$group
    fi

    if [ "$group" = "$ref_name" ] ; then
        sum=$(echo "($sum + $ovh)" | bc -l)
        count=$((count + 1))

    else
        mean=$(echo "$sum / $count" | bc -l)
        echo "$ref_name,$mean" >> $exit
        ref_name=$group
        sum=$ovh
        count=1
    fi
    
done < /tmp/nas_tmp_input.csv


if [ "$count" -gt 0 ]; then
    mean=$(echo "$sum / $count" | bc -l)
    echo "$ref_name,$mean" >> "$exit"
fi