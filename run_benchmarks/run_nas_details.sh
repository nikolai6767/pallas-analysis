#!/bin/bash

nas_dir=$PWD/run_nas_benchmark/20_iter    #MODIFY
bin_dir=$nas_dir/../NPB3.4-MPI/bin
log_dir=$nas_dir/log
traces_dir=$nas_dir/traces
details_dir=$nas_dir/details

mkdir -p $log_dir
mkdir -p $traces_dir
mkdir -p $details_dir


cd $nas_dir

NB_ITER=1
NB_RANKS=64

for i in $(seq $NB_ITER) ; do 
    for app in $bin_dir/* ; do 

        app_name=$(basename "$app")
        log_file_vanilla="$log_dir/${app_name}_${i}_vanilla.log"
        log_file_eztrace="$log_dir/${app_name}_${i}_eztrace.log"


        mpirun -np "$NB_RANKS" eztrace -m -t "mpi compiler_instrumentation" "$app" 2>&1 | tee -a "$log_file_eztrace"

        mv $nas_dir/${app_name}_trace $traces_dir/${app_name}_trace_${i}

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

        mv $PWD/pcw_enc_alg_details.csv ${details_dir}/pcw_enc_alg_details_${app_name}.csv
        mv $PWD/pcw_details.csv ${details_dir}/pcw_details_${app_name}.csv
        mv $PWD/pcw_comp_alg_details.csv ${details_dir}/pcw_comp_alg_details_${app_name}.csv
        mv $PWD/pcw_write_details.csv ${details_dir}/pcw_write_details_${app_name}.csv
        mv $PWD/write_dur_subvec_ftell_details.csv ${details_dir}/write_dur_subvec_ftell_details_${app_name}.csv
        mv $PWD/write_dur_subvec_pcw_details.csv ${details_dir}/write_dur_subvec_pcw_details_${app_name}.csv
        mv $PWD/write_dur_subvec_delete_details.csv ${details_dir}/write_dur_subvec_delete_details_${app_name}.csv

        rm $PWD/*.csv
    done
done

