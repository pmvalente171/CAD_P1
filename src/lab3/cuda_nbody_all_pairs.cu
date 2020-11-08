/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>

static constexpr int thread_block_size = 512;

namespace cadlabs {

cuda_nbody_all_pairs::cuda_nbody_all_pairs(
        const int number_particles,
        const float t_final,
        const unsigned number_of_threads,
        const universe_t universe,
        const unsigned universe_seed,
        const string file_name) :
        nbody(number_particles, t_final, universe, universe_seed, file_name),
        number_blocks ((number_particles + thread_block_size - 1)/thread_block_size)  {

    // cudaMalloc((void **)&gpu_particles, number_particles*sizeof(particle_t));
}

cuda_nbody_all_pairs::~cuda_nbody_all_pairs() {
    // cudaFree(gpu_particles);
}


__global__ void nbody_kernel(particle_t* particles, const unsigned number_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // while (true) {}

    if (index < number_particles) {

        particle_t *pi = &particles[index];
        pi->x_force = 0;
        pi->y_force = 0;

        for (int j = 0; j < number_particles; j++) {
            particle_t *pj = &particles[j];
            /* compute the force of particle j on particle i */

            double x_sep, y_sep, dist_sq, grav_base;

            x_sep = pj->x_pos - pi->x_pos;
            y_sep = pj->y_pos - pi->y_pos;
            dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

            /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
            grav_base = GRAV_CONSTANT * (pi->mass) * (pj->mass) / dist_sq;

            pi->x_force += grav_base * x_sep;
            pi->y_force += grav_base * y_sep;
        }
    }
}


/**
 * TODO: A CUDA implementation
 */
void cuda_nbody_all_pairs::calculate_forces() {
    /* First calculate force for particles. */
    cudaMalloc((void **)&gpu_particles, number_particles*sizeof(particle_t));
    uint count = number_particles * sizeof(particle_t);
    cudaMemcpy(gpu_particles, particles, count, cudaMemcpyHostToDevice);
    nbody_kernel<<<number_blocks, thread_block_size>>>(gpu_particles, number_particles);
    cudaMemcpy(particles, gpu_particles, count, cudaMemcpyDeviceToHost);
    cudaFree(gpu_particles);
}


void cuda_nbody_all_pairs::move_all_particles(double step) {
    nbody::move_all_particles(step);
}

void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
    nbody::print_all_particles(out);
}


} // namespace

