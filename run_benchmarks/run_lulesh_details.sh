#!/bin/bash

lulesh_dir=$PWD/run_lulesh
log_dir=$lulesh_dir/log
traces_dir=$lulesh_dir/traces
details_dir=$lulesh_dir/details

mkdir -p $log_dir
mkdir -p $traces_dir
mkdir -p $details_dir

cd $lulesh_dir

NB_SIM=10
NB_ITER=500
SIZE=100

cd $lulesh_dir/LULESH

for i in $(seq $NB_SIM) ; do

    log_file_vanilla="$log_dir/lulesh_${i}_vanilla.log"
    log_file_eztrace="$log_dir/lulesh_${i}_eztrace.log"

    ./lulesh2.0 -p -i $NB_ITER -s $SIZE 2>&1 | tee -a "$log_file_vanilla"

    eztrace -m -t "mpi" ./lulesh2.0 -p -i $NB_ITER -s $SIZE 2>&1 | tee -a "$log_file_eztrace"

    mv *0_trace $traces_dir/lulesh_trace_${i}

    cat $PWD/zstd.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write_vector.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write_duration_vector.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write_dur_subvec.csv | tail -n 1 >> "$log_file_eztrace"
    cat $PWD/write_subvec.csv | tail -n 1 >> "$log_file_eztrace"



        mv $PWD/zstd_details.csv ${details_dir}/zstd_details_${app_name}.csv
        mv $PWD/write_details.csv ${details_dir}/write_details_${app_name}.csv
        mv $PWD/write_vector_details.csv ${details_dir}/write_vector_details_${app_name}.csv
        mv $PWD/write_duration_vector_details.csv ${details_dir}/write_duration_vector_details_${app_name}.csv
        mv $PWD/write_dur_subvec_details.csv ${details_dir}/write_dur_subvec_details_${app_name}.csv
        mv $PWD/write_subvec_details.csv ${details_dir}/write_subvec_details_${app_name}.csv


    rm $PWD/*.csv
done
