cmake_minimum_required(VERSION 3.0.0)
project(positionning_circle_from_stereocam VERSION 0.1.0)
set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
add_compile_options(-std=c++11)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(/usr/include/eigen3/)
include_directories(${EIGEN3_INCLUDE_DIR}/eigen3)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(rough_circle_solver
  src/rough_circle_solver.cpp
)

include(CTest)
enable_testing()

link_directories(${OpenCV_LIB_DIR})
link_directories(${EIGEN3_LIBRARY_DIRS})

add_executable(demo main_demo.cpp)
target_link_libraries (demo rough_circle_solver ${OpenCV_LIBS}  ${EIGEN3_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
