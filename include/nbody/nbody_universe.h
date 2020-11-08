#ifndef NBODY_UNIVERSE_H
#define NBODY_UNIVERSE_H

#include "data_types.h"
using namespace data_types;

namespace cadlabs {

// Universes
    enum class universe_t {
        ORIGINAL,
        DISC,
        SPHERE
    };

    void original(int num_particles, particle_t *particles);

    void rotating_disc(int num_particles, particle_t *particles);

    void sphere(int num_particles, particle_t *particles);

}

#endif	/* NBODY_UNIVERSE_H */