#!/bin/sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark
file=$PWD/../res/nas_overhead.csv

touch $file

    
echo "NAME,TIME" >> $file

for app in $nas_dir/log/* ; do 

    app_name=$(basename app)
    echo $app | tr '\n' ',' >> $file
    grep -e "Time in seconds" $app | sed -e "s/Time in seconds =//g" | sed -e "s/ //g" >> $file

done