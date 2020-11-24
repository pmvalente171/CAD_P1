
#############
# NBody
#############

set (NBODY_SEQ_FOLDER ${PROJECT_SOURCE_DIR}/src/Seq)

set (NBODY_BASE_SOURCES
        ${NBODY_SEQ_FOLDER}/nbody_alloc.cpp
        ${NBODY_SEQ_FOLDER}/nbody_tools.cpp
        ${NBODY_SEQ_FOLDER}/universe_generation.cpp
        ${NBODY_SEQ_FOLDER}/nbody_all_pairs.cpp ../../include/nbody/data_types.h ../../include/nbody/data_types.h)

set (NBODY_DISPLAY_SOURCES
        ${NBODY_SEQ_FOLDER}/ui.cpp
        ${NBODY_SEQ_FOLDER}/xstuff.cpp)


set (CUDA_NBODY_ALL_PAIRS_1_CYCLE cuda_nbody_1_cycle)

if (${DISPLAY})
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ${NBODY_DISPLAY_SOURCES} ../Seq/get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)
else()
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ../Seq/get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)
endif()

add_executable(${CUDA_NBODY_ALL_PAIRS_1_CYCLE} ${NBODY_SOURCES} cuda_nbody_all_pairs.cu cuda_nbody_main.cpp)



if (${DISPLAY})
    target_compile_definitions(${CUDA_NBODY_ALL_PAIRS_1_CYCLE} PRIVATE "-DDISPLAY")
endif()

if (${DUMP_RESULT})
    target_compile_definitions(${CUDA_NBODY_ALL_PAIRS_1_CYCLE} PRIVATE "-DDUMP_RESULT")
endif()


set (LAB3_SOURCES ${NBODY_BASE_SOURCES} cuda_nbody_all_pairs.cu PARENT_SCOPE)