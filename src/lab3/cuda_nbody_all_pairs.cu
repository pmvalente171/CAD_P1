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

        hForcesX = (double *)malloc(number_particles * gridWidth * sizeof(double));
        hForcesY = (double *)malloc(number_particles * gridWidth * sizeof(double));

        cudaMalloc(&dForcesX, number_particles * gridWidth * sizeof(double));
        cudaMalloc(&dForcesY, number_particles * gridWidth * sizeof(double));
    }

    cuda_nbody_all_pairs::~cuda_nbody_all_pairs() {
        cudaFree(gpu_particles);
        free(hForcesX);
        free(hForcesY);
        cudaFree(dForcesX);
        cudaFree(dForcesY);
    }

    __global__ void calculate_forces(particle_t *particles, double *gForcesX,
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

            //printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);

            __syncthreads();


            /*
             * Reduce section
             */
            unsigned int s;
            for(s = (blockDim.x)/2; s > 32 ; s>>=1) {
                //printf("S value: %d\n", s);
                if (threadIdx.x < s) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                }
                __syncthreads();
            }

            //printf("S value: %d\n", s);
            if (threadIdx.x < s) {
                //printf("S value: %d\n", s);
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 32];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 32];
                s >>= 1;

                //printf("S value: %d\n", s);
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                //printf("S value: %d\n", s);
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                //printf("S value: %d\n", s);
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                //printf("S value: %d\n", s);
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                s >>= 1;

                //printf("S value: %d\n", s);
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
            }

            if (!threadIdx.x) {
                //printf("sForcesX[%d] corresponding to particle %d are %f\n", threadIdx.y * blockDim.x, targetParticle, sForcesX[threadIdx.y * blockDim.x]);
                gForcesX[targetParticle * gridWidth + blockIdx.x] = sForcesX[threadIdx.y * blockDim.x];
                gForcesY[targetParticle * gridWidth + blockIdx.x] = sForcesY[threadIdx.y * blockDim.x];
            }
        }
    }

    void cuda_nbody_all_pairs::calculate_forces() {
        uint size = number_particles * sizeof(particle_t);

        cudaMemcpy(gpu_particles, particles, size, cudaMemcpyHostToDevice);

        dim3 grid(gridWidth, gridHeight);
        // printf("grid dims: %d, %d\n", gridWidth, gridHeight);
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
        ::cadlabs::calculate_forces<<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
        //printf("\n\n");

        cudaMemcpy(hForcesX, dForcesX, number_particles * gridWidth * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hForcesY, dForcesY, number_particles * gridWidth * sizeof(double), cudaMemcpyDeviceToHost);

        /*for (int i = 0; i < number_particles; i++) {
            int targetParticle = i * gridWidth;
            double xF = 0; double yF = 0;
            for (int j = 0; j < gridWidth; j++) {
                printf("Particle forces : %.3f ; %.3f\n", hForcesX[targetParticle + j],
                       hForcesY[targetParticle + j]);
            }
        }*/

        for (int i = 0; i < number_particles; i++) {
            int targetParticle = i * gridWidth;
            double xF = 0; double yF = 0;
            for (int j = 0; j < gridWidth; j++) {
                // printf("index %d (i:%d, j:%d)\n", targetParticle + j, i, j);
                // printf("hForcesX[%d] = %f, for particle %d\n", i, hForcesX[targetParticle + j], targetParticle);
                // printf("hForcesX[%d] = %f, for particle %d\n", i, hForcesX[targetParticle + j], targetParticle);
                xF += hForcesX[targetParticle + j];
                yF += hForcesY[targetParticle + j];
            }
            particle_t *p = &particles[i];
            p->x_force = xF;
            p->y_force = yF;
        }
    }

    void cuda_nbody_all_pairs::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }


} // namespace
