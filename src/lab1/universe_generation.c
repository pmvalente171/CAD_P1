//
// Created by Hervé Paulino on 23/09/2020.
//


#include "nbody_universe.h"
#include <stdlib.h>
#include <math.h>

universe_t universe = ORIGINAL;

#define PI 3.14159

void original(int num_particles, particle_t *particles) {

    double total_particle = num_particles;

    for (int i = 0; i < num_particles; i++) {
        particle_t *particle = &particles[i];
#if 0
        particle->x_pos = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
        particle->y_pos = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
        particle->x_vel = particle->y_pos;
        particle->y_vel = particle->x_pos;
#else
        particle->x_pos = i * 2.0 / nparticles - 1.0;
        particle->y_pos = 0.0;
        particle->x_vel = 0.0;
        particle->y_vel = particle->x_pos;
#endif

        particle->mass = 1.0 + (num_particles + i) / total_particle;
        particle->node = NULL;
        //insert_particle(particle, root);
        printf (" x %f y %f\n", particle->x_pos, particle->y_pos);
    }
}

void sphere(int num_particles, particle_t *particles) {

    int random = (rand() * num_particles );

    double o = 2 / (double) num_particles;
    double increment = PI * (3.0 - sqrt(5));

    int rbase = MAX(1, MIN(num_particles/1000, 10));

    for (int i = 0; i < num_particles; i++) {
        particle_t *particle = &particles[i];

        double y = ((i * o) - 1) + (o / 2);
        double r = sqrt(rbase - pow(y, 2));

        double phi = ((i + random) % num_particles) * increment;

        particle->x_pos = cos(phi) * r;
        particle->y_pos = sin(phi) * r;

        particle->mass = 1.0 + (num_particles + i) / num_particles;
        particle->node = NULL;

        printf (" x %f y %f\n", particle->x_pos, particle->y_pos);
    }
}

static const float RADIUS_OFFSET = 0.05f;
static const int radius = 50;
static const float velocityMultiplier = 1.5;

void rotating_disc(int num_particles, particle_t *particles) {

    particle_t *particle = &particles[0];
    particle->mass = 1.0;
    particle->x_pos = 0;
    particle->y_pos = 0;

    for (int i = 1; i < num_particles; i++) {
        particle_t *particle = &particles[i];

        float r = (float) ((rand() * radius)  + RADIUS_OFFSET) / 100000; // + RADIUS_OFFSET to ensure it won't be on the center
        r = r - (int) r;

        double alpha = rand() * 2 * PI;
       double x = (float) (cos(alpha) * r);
        particle->x_pos = x; // - (int) x;
        double y = (float) (sin(alpha) * r);
        particle->y_pos = y; // - (int) y;

        particle->mass = 1.0 + (num_particles + i) / num_particles;

        // orbital velocity
        float v0 = (float) sqrt((particles[0].mass + particle->mass) / (r * r * r)) * velocityMultiplier;

        // rotate by 90°
        particle->x_vel = particle->y_pos * v0;
        particle->y_vel = -particle->x_pos * v0;

        particle->node = NULL;
        printf (" x %f y %f x %f y %f\n", x, y, particle->x_pos, particle->y_pos);
    }
}