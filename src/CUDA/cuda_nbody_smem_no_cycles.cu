#include <nbody/cuda_nbody_smem_no_cycles.h>
#include <omp.h>
#include "stdio.h"

static constexpr int thread_block_size = 16;
int number_blocks_width;

namespace cadlabs {

    cuda_nbody_smem_no_cycles::cuda_nbody_smem_no_cycles(
            const int number_particles,
            const float t_final,
            const unsigned number_of_threads,
            const universe_t universe,
            const unsigned universe_seed,
            const string file_name,
            const int blockWidth,
            const int blockHeight,
            const int n_streams) :
            nbody(number_particles, t_final, universe, universe_seed, file_name),
            blockWidth(blockWidth), n(n), numStreams(n_streams), blockHeight(blockHeight) {

        number_blocks_width = (number_particles + thread_block_size - 1) / thread_block_size;
        cudaMalloc((void **)&gpu_particles, number_particles*sizeof(particle_t));
    }

    cuda_nbody_smem_no_cycles::~cuda_nbody_smem_no_cycles() {
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
                p.x_force = 0;
                p.y_force = 0;
                sParticles[threadIdx.x] = p;
            }

            /*
             * All threads in a have to access each of the local arrays
             *  as such, these must be filled.
             */
            __syncthreads();

            particle_t *fp = &gParticles[forceEffectParticle];
            particle_t *tp = &sParticles[threadIdx.x];

            double x_sep = fp->x_pos - tp->x_pos;
            double y_sep = fp->y_pos - tp->y_pos;
            double dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
            double grav_base = GRAV_CONSTANT * (fp->mass) * (tp->mass) / dist_sq;

            /*
             * After computing the forces applied from one particle, these are added to the
             *  value on a local array.
             */

#ifdef GMEM_SMEM_I1
            atomicAdd(&(tp->x_force), ((float)grav_base * x_sep));
            atomicAdd(&(tp->y_force), ((float)grav_base * y_sep));
#endif
            __syncthreads();

            /*
             * Upon having computed the total forces applied to a particle in this block
             *  these are added to the global view of the particle.
             */
            if (!threadIdx.y) {

#ifdef GMEM_SMEM_I1
                atomicAdd(&(gParticles[targetParticle].x_force), tp->x_force);
                atomicAdd(&(gParticles[targetParticle].y_force), tp->y_force);
#endif
            }
        }
    }


/**
 * TODO: A CUDA implementation
 */
    void cuda_nbody_smem_no_cycles::calculate_forces() {
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
        dim3 grid(number_blocks_width, number_blocks_width);
        dim3 block(thread_block_size, thread_block_size);
        two_cycles_parallel<<<grid, block>>>(gpu_particles, number_particles);
        cudaMemcpy(particles, gpu_particles, count, cudaMemcpyDeviceToHost);
    }


    void cuda_nbody_smem_no_cycles::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_smem_no_cycles::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }


} // namespace
