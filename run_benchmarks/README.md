# Run different benchmarks

## Run NAS Parallel Benchmark

To compile NPB3.4, make sure that pallas and eztrace are already installed, then change the ```run_nas_benchmark/NPB3.4-MPI/config/make.def``` file to choose the algorithms you want.

Then, run ```bash make_nas.sh```.

To lunch the simulations, run ```bash run_nas.sh``` 