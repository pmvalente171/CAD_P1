
set(TEST_TARGET "lab2_test")

set (SOURCE_FOLDER ${PROJECT_SOURCE_DIR}/src)

set (TEST_SOURCES
        ${SOURCE_FOLDER}/SEQ/nbody_alloc.cpp
        ${SOURCE_FOLDER}/SEQ/nbody_tools.cpp
        ${SOURCE_FOLDER}/SEQ/get_output.cpp
        ${SOURCE_FOLDER}/SEQ/universe_generation.cpp
        ${SOURCE_FOLDER}/SEQ/nbody_all_pairs.cpp
        ${SOURCE_FOLDER}/PAR/par_nbody_all_pairs.cpp
        par_nbody_all_pairs_test.cpp)


add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES})

file(GLOB FILES *.log)
file(COPY ${FILES} DESTINATION ${CMAKE_BINARY_DIR}/test/lab2)