#!/bin/sh

lulesh_dir=$PWD/run_lulesh

cd $lulesh_dir

git clone git@github.com:LLNL/LULESH.git

cd LULESH/

mkdir build && cd build

cmake ..  -DCMAKE_BUILD_TYPE=Release -DMPI_CXX_COMPILER=`which mpicxx` -DCMAKE_CXX_COMPILER='/usr/bin/gcc'

cd ..

sed -i 's/mpig++/mpicxx/g' Makefile

make