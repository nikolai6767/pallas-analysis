#!/bin.sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark
file=$PWD/../res/nas_trace_size.csv

touch $file
    
echo "NOM TAILLE" >> $file
for app in $nas_dir/traces/* ; do 
    if [ -d "$app" ]; then
        app_name=$(basename app)
        du -sb $app |cut -d' ' -f1 >> $file
    fi
done