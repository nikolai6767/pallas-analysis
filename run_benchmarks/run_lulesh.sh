#!/bin/bash

lulesh_dir=$PWD/run_lulesh
log_dir=$lulesh_dir/iter_20/log
traces_dir=$lulesh_dir/iter_20/traces

mkdir -p $log_dir
mkdir -p $traces_dir

cd $lulesh_dir

NB_SIM=30
NB_ITER=500
SIZE=100

cd $lulesh_dir/LULESH

for i in $(seq $NB_SIM) ; do 

    log_file_vanilla="$log_dir/lulesh_${i}_vanilla.log"
    log_file_eztrace="$log_dir/lulesh_${i}_eztrace.log"

    /usr/bin/time -f "[TIME] %e" \
    ./lulesh2.0 -p -i $NB_ITER -s $SIZE 2>&1 | tee -a "$log_file_vanilla"

    /usr/bin/time -f "[TIME] %e" \
    eztrace -m -t "mpi" ./lulesh2.0 -p -i $NB_ITER -s $SIZE 2>&1 | tee -a "$log_file_eztrace"

    mv *0_trace $traces_dir/lulesh_trace_${i} 

    cat $PWD/zstd.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write_vector.csv | tail -n 1 >> "$log_file_eztrace"        
    cat $PWD/write_duration_vector.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write_dur_subvec.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write_subvec.csv | tail -n 1 >> "$log_file_eztrace"


    rm $PWD/*.csv   
done
