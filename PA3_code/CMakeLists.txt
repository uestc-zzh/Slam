cmake_minimum_required(VERSION 2.8)
project(drawTrajectory)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
include_directories("/usr/include/eigen3" ${Sophus_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})

add_executable(drawTrajectory draw_trajectory.cpp)
target_link_libraries(drawTrajectory ${Pangolin_LIBRARIES} Sophus::Sophus)