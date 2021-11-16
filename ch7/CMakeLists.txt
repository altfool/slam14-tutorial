cmake_minimum_required(VERSION 3.16)
project(ch7)
#set(CMAKE_CXX_STANDARD 17)
add_definitions("-DENABLE_SSE")
#message(STATUS "===${SSE_FLAGS}===")
set(CMAKE_CXX_FLAGS "-std=c++17 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


find_package(Eigen3 REQUIRED)
message(STATUS "=======EIGEN3 Include Dirs: ${EIGEN3_INCLUDE_DIRS}=========")
find_package(Sophus REQUIRED)
message(STATUS "=======SOPHUS Include Dirs: ${SOPHUS_INCLUDE_DIRS}=========")
find_package(OpenCV REQUIRED)
message(STATUS "=======OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}=========")
message(STATUS "=======OpenCV libraries: ${OpenCV_LIBRARIES}==========")
find_package(G2O REQUIRED)
SET(G2O_LIBRARIES ${G2O_STUFF_LIBRARY}
        ${G2O_CORE_LIBRARY}
        ${G2O_CLI_LIBRARY}
        ${G2O_SOLVER_CHOLMOD}
        ${G2O_SOLVER_CSPARSE}
        ${G2O_SOLVER_CSPARSE_EXTENSION}
        ${G2O_SOLVER_DENSE}
        ${G2O_SOLVER_PCG}
        ${G2O_SOLVER_SLAM2D_LINEAR}
        ${G2O_SOLVER_STRUCTURE_ONLY}
        ${G2O_SOLVER_EIGEN}
        ${G2O_TYPES_DATA}
        ${G2O_TYPES_ICP}
        ${G2O_TYPES_SBA}
        ${G2O_TYPES_SCLAM2D}
        ${G2O_TYPES_SIM3}
        ${G2O_TYPES_SLAM2D}
        ${G2O_TYPES_SLAM3D}
        )
message(STATUS "=======G2O Include Dirs: ${G2O_INCLUDE_DIRS}=========")
message(STATUS "=======G2O libraries: ${G2O_LIBRARIES}==========")
#message(STATUS "=======${G2O_CORE_LIBRARY}==========")

include_directories(${EIGEN3_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBRARIES})
add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBRARIES})
add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBRARIES})
add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBRARIES})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBRARIES} ${G2O_LIBRARIES} fmt)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBRARIES} ${G2O_LIBRARIES} fmt)