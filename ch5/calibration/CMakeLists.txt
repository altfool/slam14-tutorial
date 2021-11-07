cmake_minimum_required(VERSION 3.16)
project(ch5-cameraCalibration)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cameraCalibration cameraCalibration.cpp)
target_link_libraries(cameraCalibration ${OpenCV_LIBRARIES})