


set (NBODY_BASE_SOURCES
        nbody_alloc.c
        nbody_tools.c
        universe_generation.c
        main.c)


######################
# Find packages
######################

#### OpenMP

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
# NBody - All pairs version
##################################
link_libraries("m")
set (NBODY_ALL_PAIRS nbody)

add_executable(${NBODY_ALL_PAIRS} ${NBODY_BASE_SOURCES} nbody_all_pairs.c)

if (${DISPLAY})
    message ("-- Nbody: display option activated")
    target_compile_definitions(${NBODY_ALL_PAIRS} PRIVATE "-DDISPLAY")
endif()

if (${DUMP_RESULT})
    message ("-- Nbody: dump result")
    target_compile_definitions(${NBODY_ALL_PAIRS} PRIVATE "-DDUMP_RESULT")
endif()

##################################
# NBody - Parallel All pairs version
##################################
set (PARALLEL_NBODY_ALL_PAIRS par_nbody)

add_executable(${PARALLEL_NBODY_ALL_PAIRS} ${NBODY_BASE_SOURCES} par_nbody_all_pairs.c)
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_link_libraries(${PARALLEL_NBODY_ALL_PAIRS} /usr/local/opt/llvm/lib/libomp.dylib)
    target_compile_options(${PARALLEL_NBODY_ALL_PAIRS} PRIVATE  ${OpenMP_C_FLAGS})
else()
    target_link_libraries(${PARALLEL_NBODY_ALL_PAIRS} OpenMP::OpenMP_C)
endif()

if (${DISPLAY})
    target_compile_definitions(${PARALLEL_NBODY_ALL_PAIRS} PRIVATE "-DDISPLAY")
endif()

if (${DUMP_RESULT})
    target_compile_definitions(${PARALLEL_NBODY_ALL_PAIRS} PRIVATE "-DDUMP_RESULT")
endif()


