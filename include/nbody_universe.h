#ifndef NBODY_UNIVERSE_H
#define NBODY_UNIVERSE_H
#include "nbody.h"


// Universes
typedef enum  {
    ORIGINAL,
    DISC,
    SPHERE
} universe_t;

extern universe_t universe;

void original(int num_particles, particle_t*particles);

void rotating_disc(int num_particles, particle_t*particles);

void sphere(int num_particles, particle_t*particles);
#endif	/* NBODY_UNIVERSE_H */
