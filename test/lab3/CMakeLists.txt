
set(TEST_TARGET "lab3_test")

set (SOURCE_FOLDER ${PROJECT_SOURCE_DIR}/src)

set (CUDA_NBODY_SOURCES
        ${SOURCE_FOLDER}/lab1/nbody_alloc.cpp
        ${SOURCE_FOLDER}/lab1/nbody_tools.cpp
        ${SOURCE_FOLDER}/lab1/universe_generation.cpp
        ${SOURCE_FOLDER}/lab1/nbody_all_pairs.cpp
        ${SOURCE_FOLDER}/lab3/cuda_nbody_all_pairs.cu ../../include/nbody/data_types.h)

set (TEST_SOURCES
        vector_add_cuda_test.cu
        cuda_nbody_all_pairs_test.cu
        ${CUDA_NBODY_SOURCES})

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES})

file(GLOB FILES *.log)
file(COPY ${FILES} DESTINATION ${CMAKE_BINARY_DIR}/test/lab3)