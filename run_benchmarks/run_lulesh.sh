#!/bin/bash

lulesh_dir=$PWD/run_lulesh
log_dir=$lulesh_dir/log
traces_dir=$lulesh_dir/traces

mkdir -p $log_dir
mkdir -p $traces_dir

cd $lulesh_dir

NB_SIM=10
NB_ITER=100

cd $lulesh_dir/LULESH

for i in $(seq $NB_SIM) ; do 

    log_file_vanilla="$log_dir/lulesh_${i}_vanilla.log"
    log_file_eztrace="$log_dir/lulesh_${i}_eztrace.log"

    ./lulesh2.0 -i $NB_ITER 2>&1 | tee -a "$log_file_vanilla"

    eztrace -m -t "mpi" ./lulesh2.0 -i $NB_ITER 2>&1 | tee -a "$log_file_eztrace"

    mv $lulesh_dir/LULESH/*_trace $traces_dir/lulesh_trace_${i} 

done
