cmake_minimum_required(VERSION 2.8)
project(UseHello)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(hello ${PROJECT_SOURCE_DIR}/src/hello.cpp)
add_executable(useHello useHello.cpp)

target_link_libraries(useHello hello)

# set(CMAKE_INSTALL_PREFIX /usr/local)
# install(FILES ${PROJECT_SOURCE_DIR}/include/hello.h DESTINATION include)
# install(TARGETS hello LIBRARY DESTIANTION lib)