cmake_minimum_required(VERSION 3.16)
project(ch11)

set( CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

find_package(OpenCV REQUIRED)
message("=====OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}=====")
message("=====OpenCV libraries: ${OpenCV_LIBRARIES}=====")

find_package(DBoW3 REQUIRED)
message("=====DBoW3 Include Dirs: ${DBoW3_INCLUDE_DIRS}=====")
message("=====DBoW3 libraries: ${DBoW3_LIBRARIES}=====")

include_directories(${OpenCV_INCLUDE_DIRS} ${DBoW3_LIBRARIES})

add_executable(feature_training feature_training.cpp)
target_link_libraries(feature_training ${DBoW3_LIBRARIES})

add_executable(loop_closure loop_closure.cpp)
target_link_libraries(loop_closure ${DBoW3_LIBRARIES})