


set (NBODY_BASE_SOURCES
        nbody_alloc.c
        nbody_tools.c
        ui.c
        xstuff.c
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
# NBody - Brute force version
##################################
set (NBODY_BRUTE_FORCE nbody)

add_executable(${NBODY_BRUTE_FORCE} ${NBODY_BASE_SOURCES} nbody_brute_force.c)

if (${DISPLAY} AND ${X11_FOUND})
    message ("-- Nbody: display option activated")
    target_compile_definitions(${NBODY_BRUTE_FORCE} PRIVATE "-DDISPLAY")
endif()

if (${DUMP_RESULT})
    message ("-- Nbody: dump result")
    target_compile_definitions(${NBODY_BRUTE_FORCE} PRIVATE "-DDUMP_RESULT")
endif()

##################################
# NBody - Parallel Brute force version
##################################
set (PARALLEL_NBODY_BRUTE_FORCE par_nbody)

add_executable(${PARALLEL_NBODY_BRUTE_FORCE} ${NBODY_BASE_SOURCES} par_nbody_brute_force.c)
target_link_libraries(${PARALLEL_NBODY_BRUTE_FORCE} /usr/local/opt/llvm/lib/libomp.dylib)
target_compile_options(${PARALLEL_NBODY_BRUTE_FORCE} PRIVATE  ${OpenMP_C_FLAGS})

if (${DISPLAY} AND ${X11_FOUND})
    target_compile_definitions(${PARALLEL_NBODY_BRUTE_FORCE} PRIVATE "-DDISPLAY")
endif()

if (${DUMP_RESULT})
    target_compile_definitions(${PARALLEL_NBODY_BRUTE_FORCE} PRIVATE "-DDUMP_RESULT")
endif()


