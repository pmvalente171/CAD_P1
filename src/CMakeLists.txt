
add_executable(addarray_cuda ${CMAKE_CURRENT_SOURCE_DIR}/addarray_cuda.cu)
target_link_libraries(addarray_cuda ${GTEST_BOTH_LIBRARIES})

add_executable(card_info ${CMAKE_CURRENT_SOURCE_DIR}/card_info.cu)
target_link_libraries(card_info ${GTEST_BOTH_LIBRARIES})

add_subdirectory(wb)
