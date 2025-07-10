#!/bin.sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark
file=$PWD/../res/nas_trace_size.csv
# file2=$PWD/../res/nas_trace_length.csv

touch $file
# touch $file2


echo -n > "$file"
# echo -n > "$file2"
    
echo "SIZE,NAME" >> $file
# echo "NAME,LENGTH" >> $file2
for app in $nas_dir/traces/* ; do 
    if [ -d "$app" ]; then
        app_name=$(basename $app)
        du -sb $app | cut -d' ' -f1 | tr "\t" "," | cut -d '/' -f 1,13 | sed -e 's,/,,g'>> $file

        # echo "${app_name}," | tr -d "\n" >> $file2 
        # pallas_print ${app}/eztrace_log.pallas | wc -l >> $file2
    fi
done