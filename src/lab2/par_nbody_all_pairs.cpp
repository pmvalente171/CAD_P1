/**
 * Hervé Paulino
 */

#include <par_nbody_all_pairs.h>

namespace cadlabs {

par_nbody_all_pairs::par_nbody_all_pairs(
        const int number_particles,
        const float t_final,
        const universe_t universe,
        const unsigned number_of_threads) :

    nbody(number_particles, t_final, universe),
    number_of_threads (number_of_threads) { }


void par_nbody_all_pairs::all_move_particles(double step) {
        /* First calculate force for particles. */
    int i;

#pragma omp parallel for num_threads(number_of_threads)
    for (i = 0; i < number_particles; i++) {

        int j;
        particles[i].x_force = 0;
        particles[i].y_force = 0;
        for (j = 0; j < number_particles; j++) {
            particle_t *p = &particles[j];
            /* compute the force of particle j on particle i */
            compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
        }
    }

    /* then move all particles and return statistics */
    for (i = 0; i < number_particles; i++) {
        move_particle(&particles[i], step);
    }
}

} // namespace

