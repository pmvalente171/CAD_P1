
set(TEST_TARGET "lab3_test")

set (TEST_SOURCES vector_add_cuda_test.cu)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} ${GTEST_BOTH_LIBRARIES})
