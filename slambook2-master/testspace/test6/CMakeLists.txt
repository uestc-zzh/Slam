cmake_minimum_required(VERSION 2.8)
project(test6)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

#Eigen3
include_directories(/usr/local/include/eigen3)
# include_directories("/usr/include/eigen3")

# add_executable(${PROJECT_NAME} gaussNewton.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# add_executable(ceresCurveFitting ceresCurveFitting.cpp)
# target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
