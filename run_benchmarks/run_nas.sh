#!/bin/bash

nas_dir=$PWD/run_nas_benchmark
bin_dir=$nas_dir/NPB3.4-MPI/bin

cd $nas_dir/NPB3.4-MPI/ && make suite
cd $nas_dir

NB_RANKS=64

for i in $(seq 10) ; do 
    for app in $bin_dir/* ; do 
        mpirun -np $NB_RANKS $app 2>&1 | tee $nas_dir/log/${app}_vanilla.log
    done
done
