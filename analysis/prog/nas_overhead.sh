#!/bin/sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark
file=$PWD/../res/nas_overhead.csv
exit=$PWD/../res/overhead.csv

touch $file $exit

    
echo "NAME,TIME" >> $file

for app in $nas_dir/log/* ; do 
    echo $(basename $app) | cut -d '.' -f 1,2,3 | tr '\n' ',' >> $file
    grep -e "Time in seconds" $app | sed -e "s/Time in seconds =//g" | sed -e "s/ //g" >> $file
done


echo "NAME,DIFF" >> "$exit"
tail -n +2 "$file" | while read -r line1 && read -r line2 ; do 
    name1=$(echo "$line1" | cut -d ',' -f1)
    name2=$(echo "$line2" | cut -d ',' -f1)
    time1=$(echo "$line1" | cut -d ',' -f2)
    time2=$(echo "$line2" | cut -d ',' -f2)

    diff=$(echo "$time1 - $time2" | bc -l)
    diff=$(echo "$diff" | sed 's/^-//' )

    echo "$name1,$diff" >> $exit

done