cmake_minimum_required(VERSION 3.16)
project(ch5-stereoVision)
set(CMAKE_CXX_STANDARD 14)
message(STATUS "This is project BINARY dir " ${PROJECT_BINARY_DIR})
message(STATUS "This is project SOURCE dir " ${PROJECT_SOURCE_DIR})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBRARIES} ${EIGEN3_LIBRARIES} ${Pangolin_LIBRARIES})