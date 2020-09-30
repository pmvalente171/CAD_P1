
set (NBODY_BASE_SOURCES
        ../lab1/nbody_alloc.cpp
        ../lab1/nbody_tools.cpp
        ../lab1/universe_generation.cpp
        ../lab1/nbody_all_pairs.cpp)

set (NBODY_DISPLAY_SOURCES
        ../lab1/ui.cpp
        ../lab1/xstuff.cpp)

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
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ${NBODY_DISPLAY_SOURCES} par_nbody_all_pairs.cpp main.cpp)
else()
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES}  par_nbody_all_pairs.cpp main.cpp)
endif()

add_executable(${PARALLEL_NBODY_ALL_PAIRS} ${NBODY_SOURCES})

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


