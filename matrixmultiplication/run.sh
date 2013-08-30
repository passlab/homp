#!/bin/bash
export OMP_NUM_THREADS=16
for size in 128 256 512 1024 2048 4096; do
echo "-------------------------------- timed OpenMP ACC ${size}x${size} ------------------------------"
./matmul_time $size
echo "-------------------------------- OpenACC PGI ${size}x$size ------------------------------" 
./matmul_acc_pgi $size
echo "-------------------------------- OpenACC HMPP ${size}x$size ------------------------------" 
./matmul_acc_hmpp $size
done
