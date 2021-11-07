include_directories(/usr/include/eigen3)
add_executable(coordTrans coordinateTransform.cpp)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(plotTraj plotTrajectory.cpp)
target_link_libraries(plotTraj ${Pangolin_LIBRARIES})
