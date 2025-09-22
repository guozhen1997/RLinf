#! /bin/bash

# CUDA env
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=${CUDA_HOME}
export CUCC_PATH=${CUDA_HOME}/bin
export NVCC_PATH=${CUDA_HOME}/bin

# PATH
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}


export ARCH="NVIDIA_H100"

export MP_PP0_LAYERS=16
export UB_SKIPMC=1 
export HYDRA_FULL_ERROR=1
