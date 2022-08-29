cmake_minimum_required(VERSION 2.8)
project(EigenMatrix)

# 添加头文件
# include_directories("/usr/include/eigen3")
# 已用软连接的方式将/usr/include/eigen3/Eigen链接到了/usr/include/Eigen
add_executable(eigenMatrix eigenMatrix.cpp)
