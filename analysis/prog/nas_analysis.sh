#!/bin/sh

bash $PWD/trace_size.sh
bash $PWD/overhead.sh
bash $PWD/nas_overhead_mean.sh
bash $PWD/nas_trace_size_mean.sh
bash $PWD/nas_durations_vanilla.sh
bash $PWD/nas_durations_eztrace.sh