include_directories(/usr/include/eigen3)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(visGeo visualizeGeometry.cpp)
target_link_libraries(visGeo ${Pangolin_LIBRARIES})