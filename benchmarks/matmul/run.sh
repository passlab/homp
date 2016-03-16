#!/bin/bash

# LOOP_DIST_POLICY options: 
# 1,<n>:	BLOCK: even distribution, default n: -1
# 8,<n>:	SCHED_DYNAMIC: each dev picks chunks and chunk size does not change, default n: 100
# 9,<n>:	SCHED_GUIDED: each dev picks chunks and chunk sizes reduce each time, default n: 100
# 11,<n>:	MODEL_1_AUTO: dist all iterations using compute-based analytical model, default n: -1
# 12,<n>:	MODEL_2_AUTO: dist all iterations using compute/data-based analytical model, default n: -1
# 13,<n>:	SCHED_PROFILE_AUTO: each dev pick the same amount of chunks, runtime profiles and then dist the rest based on profiling, default n: 100
# 14,<n>:	MODEL_PROFILE_AUTO: dist the first chunk among devs using analytical model, runtime profiles, and then dist the rest based on profiling, default n: 100

export OMP_DEV_SPEC_FILE=../fornax-4gpu.ini
export LOOP_DIST_POLICY=1,10
./matmul-nvgpu-itlmic 6144

export LOOP_DIST_POLICY=8,10%
./matmul-nvgpu-itlmic 6144

export LOOP_DIST_POLICY=8,5%
./matmul-nvgpu-itlmic 6144

export LOOP_DIST_POLICY=8,2%
./matmul-nvgpu-itlmic 6144

export LOOP_DIST_POLICY=9,15%
./matmul-nvgpu-itlmic 6144

export LOOP_DIST_POLICY=11,15%
./matmul-nvgpu-itlmic 6144

export LOOP_DIST_POLICY=12,15%
./matmul-nvgpu-itlmic 6144

export LOOP_DIST_POLICY=13,5%
./matmul-nvgpu-itlmic 6144

export LOOP_DIST_POLICY=14,5%
./matmul-nvgpu-itlmic 6144
