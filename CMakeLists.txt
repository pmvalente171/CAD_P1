cmake_minimum_required(VERSION 3.10)
project(cadlabs-gpu LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

#######################
# NBody setup
#######################

set (DISPLAY FALSE)
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


##############################################################
# Tests
##############################################################

#############
# Download and unpack googletest at configure time
#############

set (GTEST_HOME ${CMAKE_SOURCE_DIR}/../googletest)

message ("-- Loading Google Test from ${GTEST_HOME}")

configure_file(${PROJECT_SOURCE_DIR}/cmake/CMakeLists.googletest.in ${GTEST_HOME}/download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${GTEST_HOME}/src
        ${GTEST_HOME}/build
        EXCLUDE_FROM_ALL)

set (GTEST_BOTH_LIBRARIES gtest_main gtest)

add_subdirectory (test)

