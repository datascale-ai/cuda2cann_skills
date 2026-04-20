cmake_minimum_required(VERSION 3.18)
project(ReduceSumLike LANGUAGES CXX CUDA)
add_library(reduce_sum_like SHARED src/binding.cpp src/reduce_sum_like.cu)
