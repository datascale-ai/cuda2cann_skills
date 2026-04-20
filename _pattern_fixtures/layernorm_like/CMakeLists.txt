cmake_minimum_required(VERSION 3.18)
project(LayerNormLike LANGUAGES CXX CUDA)
add_library(layernorm_like SHARED src/binding.cpp src/layernorm_like.cu)
