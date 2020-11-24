#include <nbody/cuda_nbody_gmem_no_cycles.h>
#include <omp.h>
#include "stdio.h"

// static constexpr int thread_block_size = 32;
int number_blocks_width;
int number_blocks_height;

namespace cadlabs {

    cuda_nbody_gmem_no_cycles::cuda_nbody_gmem_no_cycles(
            const int number_particles,
            const float t_final,
            const unsigned n,
            const universe_t universe,
            const unsigned universe_seed,
            const string file_name,
            const int blockWidth,
            const int blockHeight,
            const int n_streams ) :
            nbody(number_particles, t_final, universe, universe_seed, file_name),
            blockWidth(blockWidth), blockHeight(blockHeight), n(n), numStreams(n_streams) {

        number_blocks_width = (number_particles + blockWidth - 1) / blockWidth;
        number_blocks_height = (number_particles + blockHeight - 1) / blockHeight;
    #ifdef SOA
        // cudaMalloc((void **)&gpu_particles, number_particles*sizeof(particle_t));
        cudaMalloc((void **)&gpu_particles_soa.x_pos, number_particles*sizeof(double));
        cudaMalloc((void **)&gpu_particles_soa.y_pos, number_particles*sizeof(double));

        cudaMalloc((void **)&gpu_particles_soa.x_vel, number_particles*sizeof(double));
        cudaMalloc((void **)&gpu_particles_soa.y_vel, number_particles*sizeof(double));

        cudaMalloc((void **)&gpu_particles_soa.x_force, number_particles*sizeof(double));
        cudaMalloc((void **)&gpu_particles_soa.y_force, number_particles*sizeof(double));

        cudaMalloc((void **)&gpu_particles_soa.mass, number_particles*sizeof(double));

        // We can do this because
        // the mass of the particles
        // stays constant across the
        // whole program
        cudaMemcpy(gpu_particles_soa.mass, particles_soa.mass,
                   number_particles * sizeof(double), cudaMemcpyHostToDevice);

#else
        cudaMalloc((void **)&gpu_particles, number_particles*sizeof(particle_t));
#endif
    }

    cuda_nbody_gmem_no_cycles::~cuda_nbody_gmem_no_cycles() {
#ifdef SOA
        free(particles_soa.mass);

        free(particles_soa.x_pos);
        free(particles_soa.x_vel);
        free(particles_soa.x_force);

        free(particles_soa.y_pos);
        free(particles_soa.y_vel);
        free(particles_soa.y_force);
#else
        cudaFree(gpu_particles);
#endif
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

#ifdef ATOMIC
            atomicAdd(&(tp->x_force), grav_base * x_sep);
            atomicAdd(&(tp->y_force), grav_base * y_sep);
#endif
        }
    }

    __global__ void two_cycles_parallel_soa (
            const double * __restrict__ x_pos, const double * __restrict__ y_pos,
            force * __restrict__ x_force, force * __restrict__ y_force,
            const double * __restrict__ mass, const unsigned number_particles) {

        int targetParticle = blockIdx.x * blockDim.x + threadIdx.x;
        int forceEffectParticle = blockIdx.y * blockDim.y + threadIdx.y;

        if (targetParticle < number_particles && forceEffectParticle < number_particles) {
            // particle_t *tp = &particles[targetParticle];
            // particle_t *fp = &particles[forceEffectParticle];

            double x_sep = x_pos[forceEffectParticle] - x_pos[targetParticle];
            double y_sep = y_pos[forceEffectParticle] - y_pos[targetParticle];

            double dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
            double grav_base = GRAV_CONSTANT * (mass[forceEffectParticle]) * (mass[targetParticle]) / dist_sq;

#ifdef ATOMIC
            atomicAdd(&(x_force[targetParticle]), grav_base * x_sep);
            atomicAdd(&(y_force[targetParticle]), grav_base * y_sep);
#endif
        }
    }

#ifdef SOA
    void cuda_nbody_gmem_no_cycles::calculate_forces() {
        uint count = number_particles * sizeof(double);
        uint f_count = number_particles * sizeof(force);
        /*
         * Setting the forces to 0 within a kernel would require the synchronization of all blocks
         * An alternative solution to using the host would be to launch a kernel specifically to
         *  set the forces to 0. However, the number of particles will either not be high enough to warrant
         *  launching a kernel, or will be so high that the time to compute the forces between the
         *  particles completely eclipses the time required to set the forces to 0.
         */
#pragma omp parallel for num_threads(number_of_threads)
        for(int i = 0; i < number_particles; i++) {
            particles_soa.x_force[i] = 0;
            particles_soa.y_force[i] = 0;
        }

        cudaMemcpy(gpu_particles_soa.x_pos, particles_soa.x_pos, count, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_particles_soa.y_pos, particles_soa.y_pos, count, cudaMemcpyHostToDevice);

        cudaMemcpy(gpu_particles_soa.x_force, particles_soa.x_force, f_count, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_particles_soa.y_force, particles_soa.y_force, f_count, cudaMemcpyHostToDevice);

        dim3 grid(number_blocks_width, number_blocks_height);
        dim3 block(blockWidth, blockHeight);

        two_cycles_parallel_soa<<<grid, block>>>(
                gpu_particles_soa.x_pos, gpu_particles_soa.y_pos,
                gpu_particles_soa.x_force, gpu_particles_soa.y_force,
                gpu_particles_soa.mass, number_particles);

        cudaMemcpy(particles_soa.x_force, gpu_particles_soa.x_force, f_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(particles_soa.y_force, gpu_particles_soa.y_force, f_count, cudaMemcpyDeviceToHost);
    }
#else
    void cuda_nbody_gmem_no_cycles::calculate_forces() {
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
        dim3 grid(number_blocks_width, number_blocks_height);
        dim3 block(blockWidth, blockHeight);
        two_cycles_parallel<<<grid, block>>>(gpu_particles, number_particles);
        cudaMemcpy(particles, gpu_particles, count, cudaMemcpyDeviceToHost);
    }
#endif

    void cuda_nbody_gmem_no_cycles::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_gmem_no_cycles::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }
} // namespace
