gcc -O3 microp.c -o microp -lm -lrt

To benchmark gpu performance, use gpu-flops.cu
nvcc gpu-flops.cu -o microgpu
./microgpu <devi id>
