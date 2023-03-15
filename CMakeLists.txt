cmake_minimum_required(VERSION 3.24)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp  -O3 --std=c99")
project(lab_2_opp C)

set(CMAKE_C_STANDARD 11)

add_executable(lab_2_opp parallel_v3.c)
