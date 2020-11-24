#include <nbody/cuda_nbody_smem_no_cycles.h>
#include <omp.h>
#include "stdio.h"

static constexpr int thread_block_width = 16;
static constexpr int thread_block_height = 8;

int number_blocks_width;
int number_blocks_height;

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

        number_blocks_width = (number_particles + thread_block_width - 1) / thread_block_width;
        number_blocks_height = (number_particles + thread_block_height - 1) / thread_block_height;

#ifdef SOA
        // cudaMalloc((void **)&gpu_particles, number_particles*sizeof(particle_t));
        cudaMalloc((void **)&gpu_particles_soa.x_pos, number_particles*sizeof(double));
        cudaMalloc((void **)&gpu_particles_soa.y_pos, number_particles*sizeof(double));

        cudaMalloc((void **)&gpu_particles_soa.x_vel, number_particles*sizeof(double));
        cudaMalloc((void **)&gpu_particles_soa.y_vel, number_particles*sizeof(double));

        cudaMalloc((void **)&gpu_particles_soa.x_force, number_particles*sizeof(force));
        cudaMalloc((void **)&gpu_particles_soa.y_force, number_particles*sizeof(force));

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

    cuda_nbody_smem_no_cycles::~cuda_nbody_smem_no_cycles() {
#ifdef SOA
        cudaFree(gpu_particles_soa.mass);

        cudaFree(gpu_particles_soa.x_pos);
        cudaFree(gpu_particles_soa.x_vel);
        cudaFree(gpu_particles_soa.x_force);

        cudaFree(gpu_particles_soa.y_pos);
        cudaFree(gpu_particles_soa.y_vel);
        cudaFree(gpu_particles_soa.y_force);
#else
        cudaFree(gpu_particles);
#endif
    }

    __global__
    void two_cycles_parallel(particle_t* gParticles, const unsigned number_particles) {
        __shared__ particle_t sParticles[thread_block_width];
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

#ifdef ATOMIC
            atomicAdd(&(tp->x_force), ((float)grav_base * x_sep));
            atomicAdd(&(tp->y_force), ((float)grav_base * y_sep));
#endif
            __syncthreads();

            /*
             * Upon having computed the total forces applied to a particle in this block
             *  these are added to the global view of the particle.
             */
            if (!threadIdx.y) {

#ifdef ATOMIC
                atomicAdd(&(gParticles[targetParticle].x_force), tp->x_force);
                atomicAdd(&(gParticles[targetParticle].y_force), tp->y_force);
#endif
            }
        }
    }

    __global__ void two_cycles_parallel_soa(
            const double * x_pos, const double * y_pos,
            force * x_force, force * y_force,
            const double * mass, const unsigned number_particles) {

        __shared__ force smem_x_force[thread_block_width];
        __shared__ force smem_y_force[thread_block_width];

        __shared__ force smem_x_pos[thread_block_width];
        __shared__ force smem_y_pos[thread_block_width];

        __shared__ force smem_mass[thread_block_width];

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
                smem_x_force[threadIdx.x] = 0;
                smem_y_force[threadIdx.x] = 0;

                smem_x_pos[threadIdx.x] = x_pos[targetParticle];
                smem_y_pos[threadIdx.x] = y_pos[targetParticle];

                smem_mass[threadIdx.x] = mass[targetParticle];
            }

            /*
             * All threads in a have to access each of the local arrays
             *  as such, these must be filled.
             */
            __syncthreads();

            double x_sep = x_pos[forceEffectParticle] - smem_x_pos[threadIdx.x];
            double y_sep = y_pos[forceEffectParticle] - smem_y_pos[threadIdx.x];
            double dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
            double grav_base = GRAV_CONSTANT * (mass[forceEffectParticle]) *
                    (smem_mass[threadIdx.x]) / dist_sq;

            /*
             * After computing the forces applied from one particle, these are added to the
             *  value on a local array.
             */

#ifdef ATOMIC
            atomicAdd(&(smem_x_force[threadIdx.x]), ((float)grav_base * x_sep));
            atomicAdd(&(smem_y_force[threadIdx.x]), ((float)grav_base * y_sep));
#endif
            __syncthreads();

            /*
             * Upon having computed the total forces applied to a particle in this block
             *  these are added to the global view of the particle.
             */
            if (!threadIdx.y) {
#ifdef ATOMIC
                atomicAdd(&(x_force[targetParticle]), smem_x_force[threadIdx.x]);
                atomicAdd(&(y_force[targetParticle]), smem_y_force[threadIdx.x]);
#endif
            }
        }
    }


#ifdef SOA
    void cuda_nbody_smem_no_cycles::calculate_forces() {

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
        dim3 block(thread_block_width, thread_block_height);

        two_cycles_parallel_soa<<<grid, block>>>(
                gpu_particles_soa.x_pos, gpu_particles_soa.y_pos,
                gpu_particles_soa.x_force, gpu_particles_soa.y_force,
                gpu_particles_soa.mass, number_particles);

        cudaMemcpy(particles_soa.x_force, gpu_particles_soa.x_force, f_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(particles_soa.y_force, gpu_particles_soa.y_force, f_count, cudaMemcpyDeviceToHost);
    }
#else
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
        dim3 grid(number_blocks_width, number_blocks_height);
        dim3 block(thread_block_width, thread_block_height);
        two_cycles_parallel<<<grid, block>>>(gpu_particles, number_particles);
        cudaMemcpy(particles, gpu_particles, count, cudaMemcpyDeviceToHost);
    }
#endif

    void cuda_nbody_smem_no_cycles::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_smem_no_cycles::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }


} // namespace
