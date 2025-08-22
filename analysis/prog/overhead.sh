#!/bin/sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark/20_iter
lulesh_dir=$PWD/../../run_benchmarks/run_lulesh
file=$PWD/../res/durations.csv
exit=$PWD/../res/overhead.csv

touch $file $exit
echo -n > "$exit"
echo -n > "$file"
echo "NAME,TIME" >> $file

for app in $nas_dir/log/* ; do 
    echo $(basename $app) | cut -d '.' -f 1 | tr '\n' ',' >> $file
    grep -e "\[TIME\]" $app | sed -e "s/\[TIME\]//g" | sed -e "s/ //g" >> $file
done


# for app in $lulesh_dir/log_old/* ; do 
#     echo $(basename $app) | cut -d '.' -f1 | cut -d '_' -f 1 |tr '\n' ','>> $file
#     grep -e "Elapsed time" $app | sed -e "s/Elapsed//g" | sed -e "s/time//g" | sed -e "s/=//g" | sed -e "s/(s)//g" | sed -e "s/ //g" | awk '{printf "%.0f\n", $1}'  >> $file
# done


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