cmake_minimum_required(VERSION 3.16)
project(ch9)
set(CMAKE_CXX_FLAGS "-std=c++17")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Eigen3 REQUIRED)
message("=====Eigen3 Include Dirs: ${EIGEN3_INCLUDE_DIRS}=====")

find_package(Sophus REQUIRED)
message("=====Sophus Include Dirs: ${SOPHUS_INCLUDE_DIRS}=====")

find_package(OpenCV REQUIRED)
message("=====OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}=====")
message("=====OpenCV libraries: ${OpenCV_LIBRARIES}=====")

find_package(Ceres REQUIRED)
message("=====Ceres Include Dirs: ${CERES_INCLUDE_DIRS}=====")
message("=====Ceres libraries: ${CERES_LIBRARIES}=====")

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
message("=======G2O Include Dirs: ${G2O_INCLUDE_DIRS}=========")
message("=======G2O libraries: ${G2O_LIBRARIES}==========")

find_package(CSparse REQUIRED)
message("=======CSparse Include Dirs: ${CSPARSE_INCLUDE_DIR}=========")
message("=======CSparse libraries: ${CSPARSE_LIBRARY}==========")

include_directories(${EIGEN3_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS} ${CSPARSE_INCLUDE_DIR})

add_library(bal_common common.cpp)
add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)

target_link_libraries(bundle_adjustment_ceres bal_common ${CERES_LIBRARIES})
target_link_libraries(bundle_adjustment_g2o bal_common ${G2O_LIBRARIES} ${CSPARSE_LIBRARY} fmt)