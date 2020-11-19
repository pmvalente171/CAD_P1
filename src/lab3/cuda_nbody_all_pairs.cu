/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include "stdio.h"

static constexpr int BLOCK_WIDTH  = 256;
static constexpr int BLOCK_HEIGHT = 2;

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
        gridWidth  = number_particles / BLOCK_WIDTH + (number_particles % BLOCK_WIDTH == 0);
        gridHeight = number_particles / BLOCK_HEIGHT + (number_particles % BLOCK_HEIGHT == 0);
        hForcesX = (float **)malloc(gridHeight * sizeof(float *));
        hForcesY = (float **)malloc(gridHeight * sizeof(float *));
        cudaMalloc(&dForcesX, gridHeight * sizeof(float *));
        cudaMalloc(&dForcesY, gridHeight * sizeof(float *));
        for (int i = 0; i < gridHeight; i++) {
            hForcesX[i] = (float *) malloc(gridWidth * sizeof(float));
            hForcesY[i] = (float *) malloc(gridWidth * sizeof(float));
            cudaMalloc(&dForcesX[i], gridWidth * sizeof(float));
            cudaMalloc(&dForcesY[i], gridWidth * sizeof(float));
        }
    }

    cuda_nbody_all_pairs::~cuda_nbody_all_pairs() {
        cudaFree(gpu_particles);
        for (int i = 0; i < gridHeight; i++) {
            free(hForcesX[i]);
            free(hForcesY[i]);
            cudaFree(dForcesX[i]);
            cudaFree(dForcesY[i]);
        }
        free(hForcesX);
        free(hForcesY);
        cudaFree(dForcesX);
        cudaFree(dForcesY);
    }

    __global__ void calculate_forces(particle_t *particles, float **gForcesX,
                                     float **gForcesY, const unsigned number_particles) {
        __shared__ float sForcesX[BLOCK_HEIGHT][BLOCK_WIDTH];
        __shared__ float sForcesY[BLOCK_HEIGHT][BLOCK_WIDTH];
        int forceParticle  = blockIdx.x * blockDim.x + threadIdx.x;
        int targetParticle = blockIdx.y * blockDim.y + threadIdx.y;
        if (forceParticle < number_particles && targetParticle < number_particles) {

            /*
             * Mapping section
             */
            particle_t *fp = &particles[forceParticle];
            particle_t *tp = &particles[targetParticle];
            double x_sep = fp->x_pos - tp->x_pos;
            double y_sep = fp->y_pos - tp->y_pos;
            double dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);
            double grav_base = GRAV_CONSTANT * (fp->mass) * (tp->mass) / dist_sq;
            sForcesX[targetParticle][forceParticle] = (float)(grav_base * x_sep);
            sForcesY[targetParticle][forceParticle] = (float)(grav_base * y_sep);

            __syncthreads();

            /*
             * Reduce section
             */
            for(unsigned int s = 1; s < blockDim.x; s *= 2) {
                if (!threadIdx.x % (2 * s)) {
                    sForcesX[threadIdx.y][threadIdx.x] =
                            sForcesX[threadIdx.y][threadIdx.x + s];
                    sForcesY[threadIdx.y][threadIdx.x] =
                            sForcesX[threadIdx.y][threadIdx.x + s];
                }
                __syncthreads();
            }
            if (!threadIdx.x) {
                gForcesX[targetParticle][blockIdx.x] = sForcesX[threadIdx.y][0];
                gForcesY[targetParticle][blockIdx.x] = sForcesY[threadIdx.y][0];
            }
        }
    }

    void cuda_nbody_all_pairs::calculate_forces() {
        uint size = number_particles * sizeof(particle_t);

        cudaMemcpy(gpu_particles, particles, size, cudaMemcpyHostToDevice);

        dim3 grid(gridWidth, gridHeight);
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
        calculate_forces<<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles);

        cudaMemcpy(particles, gpu_particles, size, cudaMemcpyDeviceToHost);
    }


    void cuda_nbody_all_pairs::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }


} // namespace
