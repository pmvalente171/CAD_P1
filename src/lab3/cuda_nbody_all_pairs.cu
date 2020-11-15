/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include "stdio.h"

static constexpr int thread_block_size = 16;

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

        cudaMalloc((void **)&gpu_particles, number_particles*sizeof(particle_t));
    }

    cuda_nbody_all_pairs::~cuda_nbody_all_pairs() {
        cudaFree(gpu_particles);
    }

    __global__ void two_cycles_parallel(particle_t* gParticles, const unsigned number_particles) {
        __shared__ particle_t sParticles[thread_block_size];
        int targetParticle = blockIdx.x * blockDim.x + threadIdx.x;
        int forceEffectParticle = blockIdx.y * blockDim.y + threadIdx.y;
        if (targetParticle < number_particles && forceEffectParticle < number_particles) {

            /*
             * Only a single thread per particle per block caches the target particle to local memory
             * The particle is not inserted directly into the local array because despite the particles
             *  entering the kernel with 0 forces being applied, blocks other than this may have applied
             *  some forces
             */
            if (!threadIdx.y) {
                particle_t p = gParticles[targetParticle];
                particle_t temp;
                temp.x_pos = p.x_pos;
                temp.y_pos = p.y_pos;
                temp.x_vel = p.x_vel;
                temp.mass = p.mass;
                temp.node = p.node;
                temp.x_force = 0;
                temp.y_force = 0;
                sParticles[threadIdx.x] = temp;
            }

            /*
             * All threads in a have to access each of the local arrays
             *  as such, these must be filled.
             */
            __syncthreads();

            particle_t *tp = &sParticles[threadIdx.x];
            particle_t *fp = &gParticles[forceEffectParticle];

            double x_sep = fp->x_pos - tp->x_pos;
            double y_sep = fp->y_pos - tp->y_pos;
            double dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
            double grav_base = GRAV_CONSTANT * (fp->mass) * (tp->mass) / dist_sq;
            float forceIncreaseX = grav_base * x_sep;
            float forceIncreaseY = grav_base * y_sep;

            /*
             * After computing the forces applied from one particle, these are added to the
             *  value on a local array.
             */
            atomicAdd(&(tp->x_force), forceIncreaseX);
            atomicAdd(&(tp->y_force), forceIncreaseY);

            /*
             * Upon having computed the total forces applied to a particle in this block
             *  these are added to the global view of the particle.
             */
            if (!threadIdx.y) {
                atomicAdd(&(gParticles[targetParticle].x_force), tp->x_force);
                atomicAdd(&(gParticles[targetParticle].y_force), tp->y_force);
            }
        }
    }


/**
 * TODO: A CUDA implementation
 */
    void cuda_nbody_all_pairs::calculate_forces() {
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
    }


    void cuda_nbody_all_pairs::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }


} // namespace
