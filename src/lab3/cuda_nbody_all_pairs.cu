/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>

namespace cadlabs {

cuda_nbody_all_pairs::cuda_nbody_all_pairs(
        const int number_particles,
        const float t_final,
        const unsigned number_of_threads,
        const universe_t universe,
        const unsigned universe_seed) :

        nbody(number_particles, t_final, universe, universe_seed) {

    cudaMalloc((void **)&gpu_particles, number_particles);
}

void cuda_nbody_all_pairs::all_init_particles() {
    nbody::all_init_particles();

    // TODO cuda mem cpy to gpu particles

}

__global__ void nbody_kernel(particle_t* particles, const unsigned number_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

}

/**
 * TODO: A CUDA implementation
 */
void cuda_nbody_all_pairs::calculate_forces() {
        /* First calculate force for particles. */

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

