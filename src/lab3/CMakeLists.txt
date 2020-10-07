
add_executable(addarray_cuda ${CMAKE_CURRENT_SOURCE_DIR}/vector_add_cuda.cu)

add_executable(card_info ${CMAKE_CURRENT_SOURCE_DIR}/card_info.cu)

#############
# NBody
#############

set (NBODY_BASE_SOURCES
        ../lab1/nbody_alloc.cpp
        ../lab1/nbody_tools.cpp
        ../lab1/universe_generation.cpp
        ../lab1/nbody_all_pairs.cpp)

set (NBODY_DISPLAY_SOURCES
        ../lab1/ui.cpp
        ../lab1/xstuff.cpp)


set (CUDA_NBODY_ALL_PAIRS cuda_nbody)

if (${DISPLAY})
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ${NBODY_DISPLAY_SOURCES} cuda_nbody_all_pairs.cu main.cpp)
else()
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} par_nbody_all_pairs.cpp main.cpp)
endif()

add_executable(${PARALLEL_NBODY_ALL_PAIRS} ${NBODY_SOURCES})

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_link_libraries(${PARALLEL_NBODY_ALL_PAIRS} /usr/local/opt/llvm/lib/libomp.dylib)
    target_compile_options(${PARALLEL_NBODY_ALL_PAIRS} PRIVATE  ${OpenMP_CXX_FLAGS})
else()
    target_link_libraries(${PARALLEL_NBODY_ALL_PAIRS} OpenMP::OpenMP_CXX)
endif()

if (${DISPLAY})
    target_compile_definitions(${PARALLEL_NBODY_ALL_PAIRS} PRIVATE "-DDISPLAY")
endif()

if (${DUMP_RESULT})
    target_compile_definitions(${PARALLEL_NBODY_ALL_PAIRS} PRIVATE "-DDUMP_RESULT")
endif()


set (LAB2_SOURCES ${NBODY_BASE_SOURCES} par_nbody_all_pairs.cpp PARENT_SCOPE)