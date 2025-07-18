#!/bin/bash

nas_dir=$PWD/run_nas_benchmark
bin_dir=$nas_dir/NPB3.4-MPI/bin
log_dir=$nas_dir/log
traces_dir=$nas_dir/traces

mkdir -p $log_dir
mkdir -p $traces_dir


cd $nas_dir

NB_ITER=30
NB_RANKS=64

for i in $(seq $NB_ITER) ; do 
    for app in $bin_dir/* ; do 

        app_name=$(basename "$app")
        log_file_vanilla="$log_dir/${app_name}_${i}_vanilla.log"
        log_file_eztrace="$log_dir/${app_name}_${i}_eztrace.log"

        mpirun -np "$NB_RANKS" "$app" 2>&1 | tee -a "$log_file_vanilla"

        mpirun -np "$NB_RANKS" eztrace -m -t "mpi compiler_instrumentation" "$app" 2>&1 | tee -a "$log_file_eztrace"

        mv $nas_dir/${app_name}_trace $traces_dir/${app_name}_trace_${i}


        cat $PWD/zstd.csv | tail -n 1 >> "$log_file_eztrace"
        cat $PWD/write.csv | tail -n 1 >> "$log_file_eztrace"
        cat $PWD/write_vector.csv | tail -n 1 >> "$log_file_eztrace"
        cat $PWD/write_duration_vector.csv | tail -n 1 >> "$log_file_eztrace"
        cat $PWD/write_dur_subvec.csv | tail -n 1 >> "$log_file_eztrace"
        cat $PWD/write_subvec.csv | tail -n 1 >> "$log_file_eztrace"


        rm $PWD/*.csv
    done
done

