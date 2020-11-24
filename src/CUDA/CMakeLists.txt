
add_executable(addarray_cuda ${CMAKE_CURRENT_SOURCE_DIR}/vector_add_cuda.cu)

add_executable(card_info ${CMAKE_CURRENT_SOURCE_DIR}/card_info.cu)

#############
# NBody
#############

set (NBODY_SEQ_FOLDER ${PROJECT_SOURCE_DIR}/src/SEQ)

set (NBODY_BASE_SOURCES
        ${NBODY_SEQ_FOLDER}/nbody_alloc.cpp
        ${NBODY_SEQ_FOLDER}/nbody_tools.cpp
        ${NBODY_SEQ_FOLDER}/universe_generation.cpp
        ${NBODY_SEQ_FOLDER}/nbody_all_pairs.cpp ../../include/nbody/data_types.h ../../include/nbody/data_types.h ../../include/nbody/cuda_nbody_gmem_no_cycles.h)

set (NBODY_DISPLAY_SOURCES
        ${NBODY_SEQ_FOLDER}/ui.cpp
        ${NBODY_SEQ_FOLDER}/xstuff.cpp)


set (CUDA_NBODY_ALL_PAIRS cuda_nbody)
set (CUDA_NBODY_ALL_PAIRS_GMEM_NO_CYCLES cuda_nbody_gmem_no_cycles)
# set (CUDA_NBODY_ALL_PAIRS_FIRST cuda_nbody_gmem_first)

if (${DISPLAY})
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ${NBODY_DISPLAY_SOURCES} ../SEQ/get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)
else()
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ../SEQ/get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)
endif()

add_executable(${CUDA_NBODY_ALL_PAIRS} ${NBODY_SOURCES} cuda_nbody_all_pairs.cu cuda_nbody_main.cpp)
add_executable(${CUDA_NBODY_ALL_PAIRS_GMEM_NO_CYCLES}
                  ${NBODY_SOURCES} cuda_nbody_gmem_no_cycles.cu cuda_nbody_main_gmem_no_cycles.cpp)
# add_executable(${CUDA_NBODY_ALL_PAIRS_FIRST}
#                   ${NBODY_SOURCES} cuda_nbody_first.cu cuda_nbody_main_first.cpp)

if (${DISPLAY})
    target_compile_definitions(${CUDA_NBODY_ALL_PAIRS} PRIVATE "-DDISPLAY")
endif()

if (${DUMP_RESULT})
    target_compile_definitions(${CUDA_NBODY_ALL_PAIRS} PRIVATE "-DDUMP_RESULT")
endif()


set (LAB3_SOURCES ${NBODY_BASE_SOURCES} cuda_nbody_all_pairs.cu PARENT_SCOPE)