cmake_minimum_required(VERSION 3.0)
project(useSophus)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release" )

# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)

# Eigen
include_directories( ${Sophus_INCLUDE_DIRS} )
include_directories("/usr/include/eigen3")
add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus)

# add_subdirectory(example)
