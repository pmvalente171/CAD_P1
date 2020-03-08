cmake_minimum_required(VERSION 3.10)

set (ENV{CUDA_TOOLKIT_ROOT_DIR} "/usr/local/cuda")
set (CMAKE_CUDA_COMPILER $ENV{CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

project(cadlabs_cuda LANGUAGES CXX CUDA)

########### Cmake config

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

############ C++

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

############ CUDA

find_package(CUDA REQUIRED)


############ BUILD

include_directories(include)

add_subdirectory(src)
add_subdirectory(test)