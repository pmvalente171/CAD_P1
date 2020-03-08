enable_testing()

#############
# Download and unpack googletest at configure time
#############

set (GOOGLE_TEST_DIR ${PROJECT_SOURCE_DIR}/../googletest)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in ${GOOGLE_TEST_DIR}/googletest-download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GOOGLE_TEST_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GOOGLE_TEST_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${GOOGLE_TEST_DIR}/googletest-src
        ${GOOGLE_TEST_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

set (GTEST_BOTH_LIBRARIES gtest_main gtest)


#############
# Build tests
#############

add_subdirectory(functions)
add_subdirectory(image)

set (GTEST_BOTH_LIBRARIES ${GTEST_BOTH_LIBRARIES} PARENT_SCOPE)