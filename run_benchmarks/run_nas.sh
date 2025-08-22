#!/bin/bash

nas_dir=$PWD/run_nas_benchmark/20_iter
bin_dir=$nas_dir/../NPB3.4-MPI/bin
log_dir=$nas_dir/log
traces_dir=$nas_dir/traces

mkdir -p $log_dir
mkdir -p $traces_dir


cd $nas_dir

NB_ITER=20
NB_RANKS=64

for i in $(seq $NB_ITER) ; do 
    for app in $bin_dir/* ; do 

        app_name=$(basename "$app")
        log_file_vanilla="$log_dir/${app_name}_${i}_vanilla.log"
        log_file_eztrace="$log_dir/${app_name}_${i}_eztrace.log"
        
        /usr/bin/time -f "[TIME] %e" \
        mpirun -np "$NB_RANKS" "$app" 2>&1 | tee -a "$log_file_vanilla"

        /usr/bin/time -f "[TIME] %e" \
        mpirun -np "$NB_RANKS" eztrace -m -t "mpi compiler_instrumentation" "$app" 2>&1 | tee -a "$log_file_eztrace"

        mv $nas_dir/${app_name}_trace $traces_dir/${app_name}_trace_${i}

        done
done

