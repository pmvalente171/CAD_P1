/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_first.h>

// static constexpr int thread_block_size = 256;
int number_blocks_width = 1;

namespace cadlabs {

    cuda_nbody_first::cuda_nbody_first(
            const int number_particles,
            const float t_final,
            const unsigned number_of_threads,
            const universe_t universe,
            const unsigned universe_seed,
            const string file_name,
            const int blockWidth,
            const int n_streams) :
            nbody(number_particles, t_final, universe, universe_seed, file_name),
            blockWidth(blockWidth), n(n), numStreams(n_streams)    {

        number_blocks_width = (number_particles + blockWidth - 1) / blockWidth;

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

    cuda_nbody_first::~cuda_nbody_first() {
#ifdef SOA
        cudaFree(gpu_particles_soa.x_pos);
        cudaFree(gpu_particles_soa.y_pos);

        cudaFree(gpu_particles_soa.x_vel);
        cudaFree(gpu_particles_soa.y_vel);

        cudaFree(gpu_particles_soa.x_force);
        cudaFree(gpu_particles_soa.y_force);

        cudaFree(gpu_particles_soa.mass);
#else
        cudaFree(gpu_particles);
#endif
    }


    __global__ void nbody_kernel(particle_t* particles, const unsigned number_particles) {
        unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < number_particles) {

            particle_t *pi = &particles[index];
            pi->x_force = 0;
            pi->y_force = 0;

            for (int j = 0; j < number_particles; j++) {
                particle_t *pj = &particles[j];
                // compute the force of particle j on particle i

                double x_sep, y_sep, dist_sq, grav_base;

                x_sep = pj->x_pos - pi->x_pos;
                y_sep = pj->y_pos - pi->y_pos;
                dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

                // Use the 2-dimensional gravity rule: F = d * (GMm/d^2)
                grav_base = GRAV_CONSTANT * (pi->mass) * (pj->mass) / dist_sq;

                pi->x_force += grav_base * x_sep;
                pi->y_force += grav_base * y_sep;
            }
        }
    }

    __global__
    void nbody_kernel_soa (
            const double * __restrict__ x_pos, const double * __restrict__ y_pos,
            force * __restrict__ x_force, force * __restrict__ y_force,
            const double * __restrict__ mass, const unsigned number_particles) {

        unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < number_particles) {
            x_force[index] = 0;
            y_force[index] = 0;

            for (int j=0; j<number_particles; j++) {
                double x_sep, y_sep, dist_sq, grav_base;

                x_sep = x_pos[j] - x_pos[index];
                y_sep = y_pos[j] - y_pos[index];
                dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

                // Use the 2-dimensional gravity rule: F = d * (GMm/d^2)
                grav_base = GRAV_CONSTANT * (mass[index]) * (mass[j]) / dist_sq;

                x_force[index] += grav_base * x_sep;
                y_force[index] += grav_base * y_sep;
            }
        }
    }

/**
 * TODO: A CUDA implementation
 */

#ifdef SOA
    void cuda_nbody_first::calculate_forces() {
    // Note that in this implementation
    // we are using the whole particles
    // array in every thread, so, we
    // cannot stream parts of the array
    // to get better performance in
    // order to do this we would have to
    // change our algorithm

    uint count = number_particles * sizeof(double);
    uint f_count = number_particles * sizeof(force);
    cudaMemcpy(gpu_particles_soa.x_pos, particles_soa.x_pos, count, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_particles_soa.y_pos, particles_soa.y_pos, count, cudaMemcpyHostToDevice);

        nbody_kernel_soa<<<number_blocks_width, blockWidth>>>(
            gpu_particles_soa.x_pos, gpu_particles_soa.y_pos,
            gpu_particles_soa.x_force, gpu_particles_soa.y_force,
            gpu_particles_soa.mass, number_particles);

    cudaMemcpy(particles_soa.x_force, gpu_particles_soa.x_force, f_count, cudaMemcpyDeviceToHost);
    cudaMemcpy(particles_soa.y_force, gpu_particles_soa.y_force, f_count, cudaMemcpyDeviceToHost);
}
#else
    void cuda_nbody_first::calculate_forces() {
        uint count = number_particles * sizeof(particle_t);
        cudaMemcpy(gpu_particles, particles, count, cudaMemcpyHostToDevice);
        nbody_kernel<<<number_blocks_width, blockWidth>>>(gpu_particles, number_particles);
        cudaMemcpy(particles, gpu_particles, count, cudaMemcpyDeviceToHost);
    }
#endif

    void cuda_nbody_first::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_first::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }


} // namespace
