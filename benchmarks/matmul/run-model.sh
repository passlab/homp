#!/bin/bash
allsizes="64 128 256 512 1024 2014"

export OMP_NUM_NVGPU_DEVICES=0
export OMP_NUM_THSIM_DEVICES=1
export OMP_NUM_THREADS=32
unset OMP_NVGPU_DEVICES
for size in $allsizes; do
echo "-------------------------------------------------------------------------------------------------"
echo "-------------------------------- matmul ${size}, host run with $OMP_NUM_THREADS threads -----------------------------"
./matmul-nvgpu $size
echo "-------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------"
done

export OMP_NUM_THSIM_DEVICES=0
for dev in 0 1; do
export OMP_NVGPU_DEVICES=$dev
for size in $allsizes; do
echo "-------------------------------------------------------------------------------------------------"
echo "-------------------------------- matmul ${size}, device run on GPU $OMP_NVGPU_DEVICES -----------------------------"
./matmul-nvgpu $size
echo "-------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------"
done
done
