#!/bin.sh

lulesh_dir=$PWD/../../run_benchmarks/run_lulesh
file=$PWD/../res/lulesh_trace_size.csv

touch $file

echo "NOM TAILLE" >> $file
for app in $lulesh_dir/traces/* ; do 
    if [ -d "$app" ]; then
        app_name=$(basename app)
        du -sh $app |cut -d' ' -f1 >> $file
    fi
done