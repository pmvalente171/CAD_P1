/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>

static constexpr int thread_block_size = 256;

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

__global__ void two_cycles_parallel(particle_t* particles, const unsigned number_particles) {
    __shared__ particle_t targetParticles[thread_block_size];
    __shared__ particle_t forceParticles[thread_block_size];
    int targetParticle = blockIdx.x * blockDim.x + threadIdx.x;
    int forceEffectParticle = blockIdx.y * blockDim.y + threadIdx.y;
    if (targetParticle < number_particles && forceEffectParticle < number_particles) {
        /*
         * The particle's forces are set to 0 several times (the y dimension of the blocks)
         *  this is necessary because the synchronization is only within a block, and all blocks must
         *  view the forces initially as 0. Setting a value to 0 is concurrency free.
         * Only a single thread per particle per block caches the target particle to local memory
         */
        if (!threadIdx.y) {
            particle_t *p = &particles[targetParticle];
            p->x_force = 0;
            p->y_force = 0;
            targetParticles[threadIdx.x] = particles[targetParticle];
        }
        __syncthreads();

        if (!threadIdx.x)
            forceParticles[threadIdx.y] = particles[forceEffectParticle];
        __syncthreads();

        particle_t *tp = &targetParticles[threadIdx.x];
        particle_t *fp = &forceParticles[threadIdx.y];
        double x_sep, y_sep, dist_sq, grav_base;
        x_sep = fp->x_pos - tp->x_pos;
        y_sep = fp->y_pos - tp->y_pos;
        dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
        grav_base = GRAV_CONSTANT * (fp->mass) * (tp->mass) / dist_sq;
        double forceIncreaseX = grav_base * x_sep;
        double forceIncreaseY = grav_base * y_sep;
        atomicAdd(((float *) &(tp->x_force)), ((float)forceIncreaseX));
        atomicAdd(((float *)&(tp->y_force)), ((float)forceIncreaseY));
        __syncthreads();

        if (!threadIdx.y) {
            atomicAdd(((float *)&(particles[targetParticle].x_force)), ((float)tp->x_force));
            atomicAdd(((float *)&(particles[targetParticle].y_force)), ((float)tp->y_force));
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
    dim3 grid(number_blocks, number_blocks, 1);
    dim3 block(thread_block_size, thread_block_size, 1);
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

