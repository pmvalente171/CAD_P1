
cmake_minimum_required(VERSION 3.10)
project(cadlabs_shared_memory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(CUDA REQUIRED)

include_directories(include)

add_subdirectory(src)
add_subdirectory(test)
