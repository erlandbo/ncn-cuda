cmake_minimum_required(VERSION 3.16)
project(MyCUDACode CUDA CXX)

set(CMAKE_CUDA_COMPILER /home/erland/anaconda3/envs/ncn-cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17) # Or desired C++ standard
set(CMAKE_CUDA_ARCHITECTURES 86) # Replace with your GPU's compute capability

add_executable(${PROJECT_NAME} flash.cu)