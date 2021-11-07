cmake_minimum_required(VERSION 3.16)
project(ch2)
message(STATUS "This is project BINARY dir " ${PROJECT_BINARY_DIR})
message(STATUS "This is project SOURCE dir " ${PROJECT_SOURCE_DIR})

message(STATUS "This is cmake BINARY dir " ${CMAKE_BINARY_DIR})
message(STATUS "This is cmake SOURCE dir " ${CMAKE_SOURCE_DIR})

message(STATUS "This is cmake current BINARY dir " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "This is cmake current SOURCE dir " ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${PROJECT_BINARY_DIR}/lib)

add_executable(helloSLAM helloSLAM.cpp)
add_library(hello libHelloSLAM.cpp)
add_library(hello_shared SHARED libHelloSLAM.cpp)
add_executable(useHello useHello.cpp)
target_link_libraries(useHello hello_shared)