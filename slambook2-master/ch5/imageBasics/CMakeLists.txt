cmake_minimum_required(VERSION 2.8)
project(imageBasics)

# set(CMAKE_CXX_FLAGS "-std=c++11")

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_executable(imageBasics imageBasics.cpp)
# 链接OpenCV库
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# 链接OpenCV库
target_link_libraries(undistortImage ${OpenCV_LIBS})
