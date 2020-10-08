
set(TEST_TARGET "lab2_test")

set (SOURCE_FOLDER ${PROJECT_SOURCE_DIR}/src)

set (TEST_SOURCES
        ${SOURCE_FOLDER}/lab1/nbody_alloc.cpp
        ${SOURCE_FOLDER}/lab1/nbody_tools.cpp
        ${SOURCE_FOLDER}/lab1/universe_generation.cpp
        ${SOURCE_FOLDER}/lab1/nbody_all_pairs.cpp
        ${SOURCE_FOLDER}/lab2/par_nbody_all_pairs.cpp
        par_nbody_all_pairs_test.cpp)


add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES})

file(GLOB FILES *.log)
file(COPY ${FILES} DESTINATION ${CMAKE_BINARY_DIR}/test/lab2)