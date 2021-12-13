cmake_minimum_required(VERSION 3.16)
project(ch10)
set(CMAKE_CXX_FLAGS "-std=c++17")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Eigen3 REQUIRED)
message("=====Eigen3 Include Dirs: ${EIGEN3_INCLUDE_DIRS}=====")

find_package(Sophus REQUIRED)
message("=====Sophus Include Dirs: ${SOPHUS_INCLUDE_DIRS}=====")

find_package(G2O REQUIRED)
SET(G2O_LIBRARIES ${G2O_STUFF_LIBRARY} ${G2O_CORE_LIBRARY} ${G2O_CLI_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_SOLVER_DENSE} ${G2O_SOLVER_PCG} ${G2O_SOLVER_SLAM2D_LINEAR} ${G2O_SOLVER_STRUCTURE_ONLY} ${G2O_SOLVER_EIGEN} ${G2O_TYPES_DATA} ${G2O_TYPES_ICP} ${G2O_TYPES_SBA} ${G2O_TYPES_SCLAM2D} ${G2O_TYPES_SIM3} ${G2O_TYPES_SLAM2D} ${G2O_TYPES_SLAM3D})
message("=======G2O Include Dirs: ${G2O_INCLUDE_DIRS}=========")
message("=======G2O libraries: ${G2O_LIBRARIES}==========")

include_directories(${EIGEN3_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})

add_executable(pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3 ${G2O_LIBRARIES})

add_executable(pose_graph_g2o_lie_algebra pose_graph_g2o_lie_algebra.cpp)
target_link_libraries(pose_graph_g2o_lie_algebra ${G2O_LIBRARIES} fmt)
