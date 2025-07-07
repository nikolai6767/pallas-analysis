#!/bin/bash


base_dir=$PWD

## Creates a python virtual environment for pybind11
# mkdir .venv
# python -m venv .venv
# source .venv/bin/activate

# pip install pybind11
# export pybind11_DIR=$(python -m pybind11 --cmakedir)

## install pallas
git clone https://github.com/Pallas-Trace/pallas.git
cd pallas

## To work on the last version of pallas
git checkout dev && git pull

mkdir build install
cd build
export PALLAS_ROOT=$PWD/../install
cmake .. -DCMAKE_INSTALL_PREFIX=$PALLAS_ROOT -DENABLE_PYTHON=OFF
make -j 14 && make install


## Building EZTrace with MPI plugin
cd "$base_dir"
git clone https://gitlab.com/eztrace/eztrace.git
cd eztrace

# To work on the last version of eztrace
git checkout dev && git pull

mkdir build install
cd build
export EZTRACE_ROOT=$PWD/../install
cmake .. -DCMAKE_INSTALL_PREFIX="$EZTRACE_ROOT" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DOTF2_ROOT=$PALLAS_ROOT -DEZTRACE_ENABLE_MPI=ON
make -j 14 && make install

cd $base_dir

## To export $PATH globally
echo "export PATH=\"$EZTRACE_ROOT/bin:$PALLAS_ROOT/bin:\$PATH\"" >> $base_dir/build_pallas_eztrace/env.sh