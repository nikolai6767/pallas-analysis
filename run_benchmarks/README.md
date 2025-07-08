# Run different benchmarks

## NAS Parallel Benchmark

To compile NPB3.4, make sure that Pallas and EZTrace are already installed. Then, modify the file ```run_nas_benchmark/NPB3.4-MPI/config/make.def``` to choose the algorithms you want.

Then, run: ```bash make_nas.sh```.

To launch the simulations, run: ```bash run_nas.sh``` 

## LULESH

To build and install LULESH, run ```bash make_lulesh.sh```.
To launch the simulations, run: ```bash run_lulesh.sh
