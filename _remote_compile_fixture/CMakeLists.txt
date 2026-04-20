cmake_minimum_required(VERSION 3.18)
project(FusedBiasRelu LANGUAGES CXX CUDA)
add_library(fused_bias_relu SHARED src/binding.cpp src/fused_bias_relu.cu)
