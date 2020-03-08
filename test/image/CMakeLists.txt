set (TEST_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/image_test.u
)

include (../CMakeList.test.in)

target_link_libraries(${TEST_TARGET} libwb)