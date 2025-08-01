#!/bin/bash

nas_dir=$PWD/run_nas_benchmark/vectors
mkdir -p $nas_dir
bin_dir=$PWD/run_nas_benchmark/NPB3.4-MPI/bin

pallas_dir=$PWD/../soft/pallas/build/
patches_dir=$PWD/patches_nas

NB_RANKS=64

cd "$pallas_dir"
git checkout nikolai
git pull
cd ${nas_dir}/../

for patch_file in "$patches_dir"/*.patch; do
    patch_name=$(basename "$patch_file" .patch)

    cd "$pallas_dir"
    git apply "$patch_file" && make -j 14 && make install

    cd $nas_dir
    res_patch=${nas_dir}/${patch_name}
    mkdir $res_patch

    log_dir=$res_patch/log
    traces_dir=$res_patch/traces
    details_dir=$res_patch/details
    mkdir -p "$log_dir" "$traces_dir" "$details_dir"


    for app in $bin_dir/* ; do 

        app_name=$(basename "$app")
        log_file_eztrace="$log_dir/${app_name}_${i}_eztrace.log"

        /usr/bin/time -f "[TIME] %e\n [MAX_MEMORY] %M\n" \
        mpirun -np "$NB_RANKS" eztrace -m -t "mpi compiler_instrumentation" "$app" 2>&1 | tee -a "$log_file_eztrace"

        mv $nas_dir/${app_name}_trace $traces_dir/${app_name}_trace


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

        time=$(grep -e "\[TIME\]" $log_file_eztrace | sed -e "s/\[TIME\]//g" | sed -e "s/ //g")
        max_memory=$(grep -e "\[MAX_MEMORY\]" $log_file_eztrace | sed -e "s/\[MAX_MEMORY\]//g" | sed -e "s/ //g")

        if [ -f "perf.csv" ] && [ ! -s "perf.csv" ]; then
            echo "TIME,MAX_MEMORY" >> "perf.csv"
        fi
        echo "$time,$max_memory" >> "perf.csv

        rm $PWD/*.csv
    done

done

cd "$pallas_dir"
git reset --hard HEAD
git checkout nikolai
