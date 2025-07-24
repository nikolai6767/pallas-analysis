#!/bin.sh

lulesh_dir=$PWD/../../run_benchmarks/run_lulesh
nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark/iter_20
file=$PWD/../res/trace_size.csv

echo -n > "$file"
    
echo "SIZE,NAME" >> $file

for app in $nas_dir/traces/* ; do 
    if [ -d "$app" ]; then
        app_name=$(basename $app)
        du -sb $app | cut -d' ' -f1 | tr "\t" "," | cut -d '/' -f 1,14 | sed -e 's,/,,g' >> $file
    fi
done


for app in $lulesh_dir/traces/* ; do 
    if [ -d "$app" ]; then
        app_name=$(basename app)
        du -sb $app |cut -d' ' -f1 | tr "\t" "," | cut -d '/' -f 1,13 | sed -e 's,/,,g' | cut -d '_' -f 1 >> $file
    fi
done

# | cut -d '/' -f 1,13 | sed -e 's,/,,g'