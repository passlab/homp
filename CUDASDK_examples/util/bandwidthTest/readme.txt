Sample: Bandwidth Test
Minimum spec: SM 1.1

This is a simple test program to measure the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e.  This test application is capable of measuring device to device copy bandwidth, host to device copy bandwidth for pageable and page-locked memory, and device to host copy bandwidth for pageable and page-locked memory.

Key concepts:

The two *.cu files and the two Makefile needs to be copied to the CUDA SDK folder 1_Utilities/bandwidthTest in order to use the SDK
common and Makefile to build it and run. 
After copy the four files, "make" will build bandwidthTest executable and "make -f Makefile-parallel" will build the bandwidthTest-parallel 
executable. Launch the executable with options that can be listed using "-help" option to the executable. 
When run the bandwidthTest-parallel, use "-device=all" so it will launch multiple threads and test the bandwidth in paralle, e.g.
bandwidthTest-parallel -device=all

