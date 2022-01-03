cmake_minimum_required(VERSION 3.16)
project(dense_mono)
set(CMAKE_CXX_FLAGS "-std=c++17")


find_package(Eigen3 REQUIRED)
message("=====Eigen3 Include Dirs: ${EIGEN3_INCLUDE_DIRS}=====")

find_package(Sophus REQUIRED)
message("=====Sophus Include Dirs: ${SOPHUS_INCLUDE_DIRS}=====")

find_package(OpenCV REQUIRED)
message("=====OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}=====")
message("=====OpenCV libraries: ${OpenCV_LIBRARIES}=====")

include_directories(${EIGEN3_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(dense_mapping dense_mapping.cpp)
target_link_libraries(dense_mapping ${OpenCV_LIBRARIES} fmt)

