#!/bin/bash


nas_dir=$PWD/run_lulesh/vectors
mkdir -p $nas_dir



pallas_dir=$PWD/../soft/pallas/build/
patches_dir=$PWD/patches_nas

NB_RANKS=64


cd "$pallas_dir"
git checkout nikolai
git pull
cd ${nas_dir}/../

NB_SIM=1
NB_ITER=250
SIZE=128
app_name="lulesh"

for patch_file in "$patches_dir"/*.patch; do
    patch_name=$(basename "$patch_file" .patch)

    cd "$pallas_dir"
    git reset --hard nikolai
    git pull
    git apply "$patch_file" && make -j 14 && make install

    cd $nas_dir
    res_patch=${nas_dir}/${patch_name}
    mkdir $res_patch

    log_dir=$res_patch/log
    traces_dir=$res_patch/traces
    details_dir=$res_patch/details
    mkdir -p "$log_dir" "$traces_dir" "$details_dir"


    for i in $(seq $NB_SIM) ; do 

        log_file_eztrace="$log_dir/${app_name}_${i}_eztrace.log"

        mpirun -np 64 eztrace -m -t "mpi" ./../LULESH/lulesh2.0 -p -i $NB_ITER -s $SIZE 2>&1 | tee -a "$log_file_eztrace"

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


        rm $PWD/*.csv
    done

done



cd "$pallas_dir"
git reset --hard HEAD
git checkout nikolai
