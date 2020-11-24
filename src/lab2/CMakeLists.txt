set (NBODY_SEQ_FOLDER ${PROJECT_SOURCE_DIR}/src/Seq)

set (NBODY_BASE_SOURCES
        ${NBODY_SEQ_FOLDER}/nbody_alloc.cpp
        ${NBODY_SEQ_FOLDER}/nbody_tools.cpp
        ${NBODY_SEQ_FOLDER}/universe_generation.cpp
        ${NBODY_SEQ_FOLDER}/nbody_all_pairs.cpp ../Seq/get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)

set (NBODY_DISPLAY_SOURCES
        ${NBODY_SEQ_FOLDER}/ui.cpp
        ${NBODY_SEQ_FOLDER}/xstuff.cpp ../Seq/get_output.cpp ../../include/get_output.h)

######################
# Find packages
######################

#### OpenMP

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
    set(OpenMP_CXX_FLAGS "-fopenmp=libomp" CACHE STRING "" FORCE)
    set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
    set(OpenMP_libomp_LIBRARY "libomp" CACHE STRING "" FORCE)
    set(OpenMP_libgomp_LIBRARY "libgomp" CACHE STRING "" FORCE)
    set(OpenMP_libiomp5_LIBRARY "libiomp5" CACHE STRING "" FORCE)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(OpenMP_C "${CMAKE_C_COMPILER}" CACHE STRING "" FORCE)
    set(OpenMP_C_FLAGS "-fopenmp=libomp" CACHE STRING "" FORCE)
    set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
    set(OpenMP_libomp_LIBRARY "libomp" CACHE STRING "" FORCE)
    set(OpenMP_libgomp_LIBRARY "libgomp" CACHE STRING "" FORCE)
    set(OpenMP_libiomp5_LIBRARY "libiomp5" CACHE STRING "" FORCE)
endif()


find_package(OpenMP REQUIRED)

##################################
# NBody - Parallel All pairs version
##################################
set (PARALLEL_NBODY_ALL_PAIRS par_nbody)

if (${DISPLAY})
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ${NBODY_DISPLAY_SOURCES} ../Seq/get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)
else()
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ../../include/nbody/data_types.h ../../include/nbody/data_types.h)
endif()

add_executable(${PARALLEL_NBODY_ALL_PAIRS} ${NBODY_SOURCES} par_nbody_all_pairs.cpp main.cpp)

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