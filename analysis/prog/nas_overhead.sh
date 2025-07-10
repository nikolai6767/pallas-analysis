#!/bin/sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark
file=$PWD/../res/nas_durations.csv
exit=$PWD/../res/nas_overhead.csv

touch $file $exit
echo -n > "$exit"
echo -n > "$file"
echo "NAME,TIME" >> $file

for app in $nas_dir/log/* ; do 
    echo $(basename $app) | cut -d '.' -f 1 | tr '\n' ',' >> $file
    grep -e "Time in seconds" $app | sed -e "s/Time in seconds =//g" | sed -e "s/ //g" >> $file
done


echo "NAME,DIFF" >> "$exit"
lines=()
while IFS= read -r line; do
    lines+=("$line")
    if [ ${#lines[@]} -eq 2 ]; then
        name1=$(echo "${lines[0]}" | cut -d ',' -f1)
        time1=$(echo "${lines[0]}" | cut -d ',' -f2)
        time2=$(echo "${lines[1]}" | cut -d ',' -f2)
        diff=$(echo "$time1 - $time2" | bc -l)
        diff=$(echo "$diff" | sed 's/^-//')
        echo "$name1,$diff" >> "$exit"
        lines=()
    fi
done < <(tail -n +2 "$file")

