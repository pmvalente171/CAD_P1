
set (NBODY_BASE_SOURCES
        nbody_alloc.cpp
        nbody_tools.cpp
        universe_generation.cpp
        nbody_all_pairs.cpp get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)

set (NBODY_DISPLAY_SOURCES
        ui.cpp
        xstuff.cpp)

##################################
# NBody - All pairs version
##################################
link_libraries("m")

set (NBODY_ALL_PAIRS nbody)

if (${DISPLAY})
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} ${NBODY_DISPLAY_SOURCES} get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)
else()
    set (NBODY_SOURCES ${NBODY_BASE_SOURCES} get_output.cpp ../../include/get_output.h ../../include/nbody/data_types.h)
endif()

add_executable(${NBODY_ALL_PAIRS} ${NBODY_SOURCES} main.cpp)

if (${DISPLAY})
    message ("-- Nbody: display option activated")
    target_compile_definitions(${NBODY_ALL_PAIRS} PRIVATE "-DDISPLAY")
endif()

if (${DUMP_RESULT})
    message ("-- Nbody: dump result")
    target_compile_definitions(${NBODY_ALL_PAIRS} PRIVATE "-DDUMP_RESULT")
endif()


