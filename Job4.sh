#!/bin/bash

# == Set SGE options:

#$ -S /bin/bash
#$ -A ferenc
#$ -cwd
#$ -q sunna_gpu.q

# export LD_LIBRARY_PATH=/usr/local/cuda-5.0/lib64:$LD_LIBRARY_PATH

#nvidia-smi
./Job4.exe