/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include <stdio.h>

static constexpr int BLOCK_WIDTH  = 256;
static constexpr int BLOCK_HEIGHT = 1;

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

        cudaMalloc(&gpu_particles, number_particles*sizeof(particle_t));
        gridWidth  = number_particles / (BLOCK_WIDTH * 2) + (number_particles % (BLOCK_WIDTH * 2) != 0);
        gridHeight = number_particles / (BLOCK_HEIGHT) + (number_particles % (BLOCK_HEIGHT) != 0);

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

        hForcesX = (double *)malloc(number_particles * gridWidth * sizeof(double));
        hForcesY = (double *)malloc(number_particles * gridWidth * sizeof(double));
        cudaMalloc(&dForcesX, number_particles * gridWidth * sizeof(double));
        cudaMalloc(&dForcesY, number_particles * gridWidth * sizeof(double));
    }

    cuda_nbody_all_pairs::~cuda_nbody_all_pairs() {

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
        // cudaFree(gpu_particles);
        free(hForcesX);
        free(hForcesY);
        cudaFree(dForcesX);
        cudaFree(dForcesY);
    }

    __global__ void calculate_force_aos(
            particle_t *particles, double *gForcesX,
            double *gForcesY, const unsigned number_particles,
            const unsigned gridWidth) {

        __shared__ double sForcesX[BLOCK_HEIGHT * BLOCK_WIDTH];
        __shared__ double sForcesY[BLOCK_HEIGHT * BLOCK_WIDTH];

        unsigned int forceParticle  = blockIdx.x * 2 * blockDim.x + threadIdx.x;
        unsigned int targetParticle = blockIdx.y * blockDim.y + threadIdx.y;

        sForcesX[threadIdx.y * blockDim.x + threadIdx.x] = .0;
        sForcesY[threadIdx.y * blockDim.x + threadIdx.x] = .0;

        if (forceParticle < number_particles
            && targetParticle < number_particles) {
            /*
             * Mapping section
             */
            int b = ((forceParticle + blockDim.x) < number_particles);

            particle_t *fp_1 = &particles[forceParticle], *fp_2 = &particles[forceParticle + blockDim.x];
            particle_t *tp = &particles[targetParticle];

            double x_sep_1 = fp_1->x_pos - tp->x_pos, x_sep_2 = fp_2->x_pos - tp->x_pos;
            double y_sep_1 = fp_1->y_pos - tp->y_pos, y_sep_2 = fp_2->y_pos - tp->y_pos;

            double dist_sq_1 = MAX((x_sep_1 * x_sep_1) + (y_sep_1 * y_sep_1), 0.01);
            double dist_sq_2 = MAX((x_sep_2 * x_sep_2) + (y_sep_2 * y_sep_2), 0.01);

            double grav_base_1 = GRAV_CONSTANT * (fp_1->mass) * (tp->mass) / dist_sq_1;
            double grav_base_2 = GRAV_CONSTANT * (fp_2->mass) * (tp->mass) / dist_sq_2;

            sForcesX[threadIdx.y * blockDim.x + threadIdx.x] =
                    grav_base_1 * x_sep_1 + b * (grav_base_2 * x_sep_2);
            sForcesY[threadIdx.y * blockDim.x + threadIdx.x] =
                    grav_base_1 * y_sep_1 + b * (grav_base_2 * y_sep_2);

            __syncthreads();

            /*
             * Reduce section
             */
            unsigned int s;
            for(s = (blockDim.x)/2; s > 32 ; s>>=1) {
                if (threadIdx.x < s) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                }
                __syncthreads();
            }

            if (threadIdx.x < s) {
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 32];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 32];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
            }
            if (!threadIdx.x) {
                gForcesX[targetParticle * gridWidth + blockIdx.x] = sForcesX[threadIdx.y * blockDim.x];
                gForcesY[targetParticle * gridWidth + blockIdx.x] = sForcesY[threadIdx.y * blockDim.x];
            }
        }
    }

    __global__ void calculate_forces_soa(
            const double * __restrict__ x_pos, const double * __restrict__ y_pos,
            double * __restrict__ gForcesX, double * __restrict__ gForcesY,
            const double * __restrict__ mass,
            const unsigned number_particles,
            const unsigned gridWidth) {

        __shared__ double sForcesX[BLOCK_HEIGHT * BLOCK_WIDTH];
        __shared__ double sForcesY[BLOCK_HEIGHT * BLOCK_WIDTH];

        unsigned int forceParticle  = blockIdx.x * 2 * blockDim.x + threadIdx.x;
        unsigned int targetParticle = blockIdx.y * blockDim.y + threadIdx.y;

        sForcesX[threadIdx.y * blockDim.x + threadIdx.x] = .0;
        sForcesY[threadIdx.y * blockDim.x + threadIdx.x] = .0;

        if (forceParticle < number_particles
            && targetParticle < number_particles) {
            /*
             * Mapping section
             */
            int b = ((forceParticle + blockDim.x) < number_particles);

            double x_sep_1 = x_pos[forceParticle] - x_pos[targetParticle],
                    x_sep_2 = x_pos[forceParticle + blockDim.x] - x_pos[targetParticle];
            double y_sep_1 = y_pos[forceParticle] - y_pos[targetParticle],
                    y_sep_2 = y_pos[forceParticle + blockDim.x] - y_pos[targetParticle];

            double dist_sq_1 = MAX((x_sep_1 * x_sep_1) + (y_sep_1 * y_sep_1), 0.01);
            double dist_sq_2 = MAX((x_sep_2 * x_sep_2) + (y_sep_2 * y_sep_2), 0.01);

            double grav_base_1 = GRAV_CONSTANT * (mass[forceParticle])
                                 * (mass[targetParticle]) / dist_sq_1;
            double grav_base_2 = GRAV_CONSTANT * (mass[forceParticle + blockDim.x])
                                 * (mass[targetParticle]) / dist_sq_2;

            sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                    (grav_base_1 * x_sep_1) + b * (grav_base_2 * x_sep_2);
            sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                    (grav_base_1 * y_sep_1) + b * (grav_base_2 * y_sep_2);

            __syncthreads();

            /*
             * Reduce section
             */
            unsigned int s;
            for(s = (blockDim.x)/2; s > 32 ; s>>=1) {
                if (threadIdx.x < s) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                }
                __syncthreads();
            }

            if (threadIdx.x < s) {
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 32];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 32];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
            }
            if (!threadIdx.x) {
                gForcesX[targetParticle * gridWidth + blockIdx.x] = sForcesX[threadIdx.y * blockDim.x];
                gForcesY[targetParticle * gridWidth + blockIdx.x] = sForcesY[threadIdx.y * blockDim.x];
            }
        }
    }

#ifdef SOA
    void cuda_nbody_all_pairs::calculate_forces() {
        uint count = number_particles * sizeof(double);
        dim3 grid(gridWidth, gridHeight);
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);

        cudaMemcpy(gpu_particles_soa.x_pos, particles_soa.x_pos, count, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_particles_soa.y_pos, particles_soa.y_pos, count, cudaMemcpyHostToDevice);
        calculate_forces_soa<<<grid, block>>>(gpu_particles_soa.x_pos, gpu_particles_soa.y_pos, dForcesX, dForcesY,
                                              gpu_particles_soa.mass, number_particles, gridWidth);
        cudaMemcpy(hForcesX, dForcesX, number_particles * gridWidth * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hForcesY, dForcesY, number_particles * gridWidth * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < number_particles; i++) {
            int targetParticle = i * gridWidth;
            double xF = 0; double yF = 0;
            for (int j = 0; j < gridWidth; j++) {
                xF += hForcesX[targetParticle + j];
                yF += hForcesY[targetParticle + j];
            }
            particles_soa.x_force[i] = xF;
            particles_soa.y_force[i] = yF;
        }
    }
#else
    void cuda_nbody_all_pairs::calculate_forces() {
        uint size = number_particles * sizeof(particle_t);
        dim3 grid(gridWidth, gridHeight);
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);

        cudaMemcpy(gpu_particles, particles, size, cudaMemcpyHostToDevice);
        calculate_force_aos<<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
        cudaMemcpy(hForcesX, dForcesX, number_particles * gridWidth * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hForcesY, dForcesY, number_particles * gridWidth * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < number_particles; i++) {
            int targetParticle = i * gridWidth;
            double xF = 0; double yF = 0;
            for (int j = 0; j < gridWidth; j++) {
                xF += hForcesX[targetParticle + j];
                yF += hForcesY[targetParticle + j];
            }
            particle_t *p = &particles[i];
            p->x_force = xF;
            p->y_force = yF;
        }
    }
#endif

    void cuda_nbody_all_pairs::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }
} // namespace
