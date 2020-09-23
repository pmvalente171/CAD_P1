cmake_minimum_required(VERSION 3.17)
project(cadlabs-gpu LANGUAGES C)

set(CMAKE_C_STANDARD 11)

#######################
# NBody setup
#######################

set (LINE_UNIVERSE 0)
set (DISC_UNIVERSE 1)

set (DISPLAY true)
set (DUMP_RESULT false)

#######################
# Find packages
#######################

#### X11

find_package(X11)
if (X11_FOUND)
    include_directories(${X11_INCLUDE_DIR})
    link_libraries(${X11_LIBRARIES})
else()
    message ("X11 not found - no graphical display support")
endif()


include_directories(include)
add_subdirectory(src)
