/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include "stdio.h"

static constexpr int thread_block_size = 32;

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

__global__ void two_cycles_parallel(particle_t* particles, const unsigned number_particles) {
    int targetParticle = blockIdx.x * blockDim.x + threadIdx.x;
    int forceEffectParticle = blockIdx.y * blockDim.y + threadIdx.y;
    if (targetParticle < number_particles && forceEffectParticle < number_particles) {
        particle_t *tp = &particles[targetParticle];
        particle_t *fp = &particles[forceEffectParticle];
        double x_sep = fp->x_pos - tp->x_pos;
        double y_sep = fp->y_pos - tp->y_pos;
        double dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
        double grav_base = GRAV_CONSTANT * (fp->mass) * (tp->mass) / dist_sq;
        atomicAdd(&(tp->x_force), grav_base * x_sep);
        atomicAdd(&(tp->y_force), grav_base * y_sep);
    }
}

/**
 * TODO: A CUDA implementation
 */
void cuda_nbody_all_pairs::calculate_forces() {
    cudaMalloc((void **)&gpu_particles, number_particles*sizeof(particle_t));
    uint count = number_particles * sizeof(particle_t);

    /*
     * Setting the forces to 0 within a kernel would require the synchronization of all blocks
     * An alternative solution to using the host would be to launch a kernel specifically to
     *  set the forces to 0. However, the number of particles will either not be high enough to warrant
     *  launching a kernel, or will be so high that the time to compute the forces between the
     *  particles completely eclipses the time required to set the forces to 0.
     */
    #pragma omp parallel for num_threads(number_of_threads)
    for(int i = 0; i < number_particles; i++) {
        particle_t* p = &particles[i];
        p->x_force = 0;
        p->y_force = 0;
    }

    cudaMemcpy(gpu_particles, particles, count, cudaMemcpyHostToDevice);
    dim3 grid(number_blocks, number_blocks);
    dim3 block(thread_block_size, thread_block_size);
    two_cycles_parallel<<<grid, block>>>(gpu_particles, number_particles);
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

