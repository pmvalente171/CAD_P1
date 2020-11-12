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


/*__global__ void nbody_kernel(particle_t* particles, const unsigned number_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < number_particles) {
        particle_t *pi = &particles[index];
        pi->x_force = 0;
        pi->y_force = 0;
        for (int j = 0; j < number_particles; j++) {
            particle_t *pj = &particles[j];
            /* compute the force of particle j on particle i */
   /*         double x_sep, y_sep, dist_sq, grav_base;
            x_sep = pj->x_pos - pi->x_pos;
            y_sep = pj->y_pos - pi->y_pos;
            dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
            /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
       /*     grav_base = GRAV_CONSTANT * (pi->mass) * (pj->mass) / dist_sq;
            pi->x_force += grav_base * x_sep;
            pi->y_force += grav_base * y_sep;
        }
    }
}*/

__global__ void two_cycles_parallel(particle_t* particles, const unsigned number_particles) {
    __shared__ particle_t targetParticles[thread_block_size];
    __shared__ particle_t forceParticles[thread_block_size];
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
            //targetParticles[threadIdx.x] = particles[targetParticle];
            particle_t p = particles[targetParticle];
            particle_t temp;
            temp.x_pos = p.x_pos;
            temp.y_pos = p.y_pos;
            temp.x_vel = p.x_vel;
            temp.mass = p.mass;
            temp.node = p.node;
            temp.x_force = 0;
            temp.y_force = 0;
            targetParticles[threadIdx.x] = temp;
            //printf("tempPos: (%f, %f)\n", temp.x_pos, temp.y_pos);
            //printf("tempMass: %f\n", temp.mass);
        }

        /*
         * Same comment for this block of code
         */
        if (!threadIdx.x) {
            //forceParticles[threadIdx.y] = particles[forceEffectParticle];
            particle_t p = particles[forceEffectParticle];
            particle_t temp;
            temp.x_pos = p.x_pos;
            temp.y_pos = p.y_pos;
            temp.x_vel = p.x_vel;
            temp.mass = p.mass;
            temp.node = p.node;
            temp.x_force = 0;
            temp.y_force = 0;
            forceParticles[threadIdx.y] = temp;
        }

        /*
         * All threads in a have to access each of the local arrays
         *  as such, these must be filled.
         */
         __syncthreads();

        /*if (!threadIdx.x && !threadIdx.y)
            //    printf("Here\n");
            for(int i = 0; i < number_particles; i++)
                printf("forces: (%f, %f)\n"
                       "thread: (%d, %d)\n",
                       targetParticles[i].x_force, targetParticles[i].y_force,
                       targetParticle, forceEffectParticle);
        __syncthreads();*/

        particle_t *tp = &targetParticles[threadIdx.x];
        particle_t *fp = &forceParticles[threadIdx.y];

        double x_sep = fp->x_pos - tp->x_pos;
        double y_sep = fp->y_pos - tp->y_pos;
        //printf("displacement: (%f, %f)\n", x_sep, y_sep);
        double dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
        //printf("distSq: %f\n", dist_sq);
        double grav_base = GRAV_CONSTANT; //* (fp->mass) * (tp->mass) / dist_sq;
        //printf("Mass particles: %f, %f  \n", fp->mass, tp->mass);
        //printf("grav: %f\n", grav_base);
        float forceIncreaseX = grav_base * x_sep;
        float forceIncreaseY = grav_base * y_sep;
        //printf("force increase: (%f, %f)\n", forceIncreaseX, forceIncreaseY);

        /*
         * After computing the forces applied from one particle, these are added to the
         *  value on a local array.
         * TODO The order the threadIdx are read may have to be altered to ensure
         *  calls to these atomicAdd are more spaced.
         */
        atomicAdd(&(tp->x_force), forceIncreaseX);
        atomicAdd(&(tp->y_force), forceIncreaseY);

        __syncthreads();

       /* if (!threadIdx.x && !threadIdx.y)
            //    printf("Here\n");
            for(int i = 0; i < number_particles; i++)
                printf("forces: (%f, %f)\n"
                       "thread: (%d, %d)\n",
                       targetParticles[i].x_force, targetParticles[i].y_force,
                       targetParticle, forceEffectParticle);
        __syncthreads();*/

        /*
         * Upon having computed the total forces applied to a particle in this block
         *  these are added to the global view of the particle.
         */
        if (!threadIdx.y) {
            atomicAdd(&(particles[targetParticle].x_force), tp->x_force);
            atomicAdd(&(particles[targetParticle].y_force), tp->y_force);
        }
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
    //#pragma omp parallel for num_threads(number_of_threads)
    for(int i = 0; i < number_particles; i++) {
        particle_t* p = &particles[i];
        p->x_force = 0;
        p->y_force = 0;
    }

    cudaMemcpy(gpu_particles, particles, count, cudaMemcpyHostToDevice);
    dim3 grid(number_blocks, number_blocks);
    dim3 block(thread_block_size, thread_block_size);
    //printf("number blocks: %d\n", number_blocks);
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

