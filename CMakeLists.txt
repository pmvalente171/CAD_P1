cmake_minimum_required(VERSION 3.10)
project(cadlabs-gpu LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

#######################
# NBody setup
#######################

set (DISPLAY TRUE)
set (DUMP_RESULT FALSE)

#######################
# Find packages
#######################

#### X11

if (${DISPLAY})

    find_package(X11 REQUIRED)

    if (X11_FOUND)
        include_directories(${X11_INCLUDE_DIR})
        link_libraries(${X11_LIBRARIES})
    else()
        message ("X11 not found - no graphical display support")
    endif()

endif()

include_directories(include)
add_subdirectory(src)
