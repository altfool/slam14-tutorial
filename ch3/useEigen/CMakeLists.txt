cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")

include_directories("/usr/include/eigen3")
add_executable(eigen eigenMatrix.cpp)