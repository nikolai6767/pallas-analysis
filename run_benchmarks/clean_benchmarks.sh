#!/bin/sh

src=$PWD
nas_src=$src/run_nas_benchmark/NPB3.4-MPI
lulesh_src=$src/run_lulesh/LULESH

cd $nas_src
make clean
rm -f bin/*

cd $lulesh_src
make clean
rm -rf build