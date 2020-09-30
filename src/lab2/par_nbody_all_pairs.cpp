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


void par_nbody_all_pairs::calculate_forces() {
        /* First calculate force for particles. */

#pragma omp parallel for num_threads(number_of_threads)
    for (int i = 0; i < number_particles; i++) {

        particles[i].x_force = 0;
        particles[i].y_force = 0;
        for (int j = 0; j < number_particles; j++) {
            particle_t *p = &particles[j];
            /* compute the force of particle j on particle i */
            compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
        }
    }
}



} // namespace

