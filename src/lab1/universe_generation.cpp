//
// Created by Hervé Paulino on 23/09/2020.
//

#include "nbody/nbody_universe.h"

#include <stdlib.h>
#include <math.h>

#include "nbody.h"

namespace cadlabs {

    universe_t universe = universe_t::ORIGINAL;

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
            particle->x_pos = i * 2.0 / num_particles - 1.0;
            particle->y_pos = 0.0;
            particle->x_vel = 0.0;
            particle->y_vel = particle->x_pos;
#endif

            particle->mass = 1.0 + (num_particles + i) / total_particle;
            particle->node = NULL;
            //insert_particle(particle, root);
        }
    }

    void original(
            int num_particles,
            double *mass,
            double *x_pos, double *y_pos,
            double *x_vel, double *y_vel) {

        double total_particle = num_particles;
        for (int i = 0; i < num_particles; i++) {
#if 0
            x_pos[i] = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
            y_pos[i] = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
            x_vel[i] = y_pos[i];
            y_vel[i] = x_pos[i];
#else
            x_pos[i] = i * (2.0 / num_particles) - 1.0f;
            y_pos[i] = 0.0;
            x_vel[i] = 0.0;
            y_vel[i] = x_pos[i];
#endif
            mass[i] = 1.0 + (num_particles + i) / total_particle;
        }
    }

    void sphere(int num_particles, particle_t *particles) {

        int random = (rand() * num_particles);

        double o = 2 / (double) num_particles;
        double increment = PI * (3.0 - sqrt(5));

        int rbase = MAX(1, MIN(num_particles / 1000, 10));

        for (int i = 0; i < num_particles; i++) {
            particle_t *particle = &particles[i];

            double y = ((i * o) - 1) + (o / 2);
            double r = sqrt(rbase - pow(y, 2));

            double phi = ((i + random) % num_particles) * increment;

            particle->x_pos = cos(phi) * r;
            particle->y_pos = sin(phi) * r;

            particle->mass = 1.0 + (num_particles + i) / num_particles;
            particle->node = NULL;

        }
    }

    void sphere(
            int num_particles,
            double *mass,
            double *x_pos, double *y_pos) {

        int random = (rand() * num_particles);
        double o = 2 / (double) num_particles;
        double increment = PI * (3.0 - sqrt(5));

        int rbase = MAX(1, MIN(num_particles / 1000, 10));

        for (int i = 0; i < num_particles; i++) {
            double y = ((i * o) - 1) + (o / 2);
            double r = sqrt( rbase - pow(y, 2.0f));
            double phi = ((i + random) % num_particles) * increment;

            x_pos[i] = cos(phi) * r;
            y_pos[i] = sin(phi) * r;

            mass[i] = 1.0f + (num_particles + i) / num_particles;
        }
    }

    void rotating_disc(int num_particles, particle_t *particles) {

        static const float RADIUS_OFFSET = 0.05f;
        static const float velocityMultiplier = 1.3;
        static const int radius = 5;

        particle_t *particle = &particles[0];
        particle->mass = 1.0;
        particle->x_pos = 0;
        particle->y_pos = 0;
        particle->x_vel = 1;


        for (int i = 0; i < num_particles; i++) {
            particle_t *particle = &particles[i];

            double r = rand() / RAND_MAX * radius + RADIUS_OFFSET;

            double alpha = rand() * 2 * PI;

            particle->x_pos = (float) (cos(alpha) * r);
            particle->y_pos = (float) (sin(alpha) * r);

            particle->mass = 1.0 + (num_particles + i) / num_particles;

            // orbital velocity
            float v0 = (float) sqrt((1 + particle->mass) / (r * r * r)) * velocityMultiplier;

            // rotate by 90°
            particle->x_vel = particle->y_pos * v0;
            particle->y_vel = -particle->x_pos * v0;

            particle->node = NULL;
            //  printf (" x %f y %f x %f y %f\n", x, y, particle->x_pos, particle->y_pos);
        }
    }

    void rotating_disc(
            int num_particles,
            double *mass,
            double *x_pos, double *y_pos,
            double *x_vel, double *y_vel) {

        static const float RADIUS_OFFSET = 0.05f;
        static const float velocityMultiplier = 1.3;
        static const int radius = 5;

        mass[0] = 1.0;
        x_pos[0] = 0;
        y_pos[0] = 0;
        x_vel[0] = 1;

        for (int i = 0; i < num_particles; i++) {
            double r = rand() / RAND_MAX * radius + RADIUS_OFFSET;
            double alpha = rand() * 2 * PI;

            x_pos[i] = (float) (cos(alpha) * r);
            y_pos[i] = (float) (sin(alpha) * r);
            mass[i] = 1.0f + (num_particles + i) / num_particles;

            // orbital velocity
            double v0 = sqrt((1 + mass[i]) / (r * r * r)) * velocityMultiplier;

            // rotate by 90°
            x_vel[i] = y_pos[i] * v0;
            y_vel[i] = -x_pos[i] * v0;
        }
    }
}