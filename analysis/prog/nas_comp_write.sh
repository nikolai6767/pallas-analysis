#!/bin/sh

nas_dir=$PWD/../../run_benchmarks/run_nas_benchmark/20_iter
log_dir=$nas_dir/log
exit=$PWD/../res/nas_comp_write.csv
tmp=$PWD/temp.csv

touch $exit
echo -n > "$exit"
echo -n > "$tmp"


echo "source,algo,n_calls,total_time,min,max,mean" > "$exit"

for file in "$log_dir"/*; do
    name=$(basename $file | cut -d '.' -f 1 )


    grep -E '^(zstd|write|write_vector|write_duration_vector|write_dur_subvec|write_subvec)' "$file" >> $tmp

    cat $tmp | while IFS=',' read -r a b c d e f; do
        echo "$name,$a,$b,$c,$d,$e,$f" >> $exit
    done

    rm $tmp
done

