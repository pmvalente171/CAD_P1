/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include <stdio.h>

static constexpr int BLOCK_WIDTH  = 256;
static constexpr int BLOCK_HEIGHT = 2;
static constexpr int n = 2;

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

        cudaMalloc(&gpu_particles, number_particles*sizeof(particle_t));
        gridWidth  = number_particles / (BLOCK_WIDTH * 2 * n) + (number_particles % (BLOCK_WIDTH * 2 * n) != 0);
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

    template<unsigned int blockSize>
    __global__ void calculate_forces(particle_t *particles, double *gForcesX,
                                     double *gForcesY, const unsigned int number_particles,
                                     const unsigned int gridWidth, const unsigned int n) {

        __shared__ double sForcesX[BLOCK_HEIGHT * BLOCK_WIDTH];
        __shared__ double sForcesY[BLOCK_HEIGHT * BLOCK_WIDTH];

        unsigned int forceParticle  = blockIdx.x * 2 * blockDim.x + threadIdx.x;
        unsigned int targetParticle = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int gridSize = blockDim.x * 2 * gridDim.x, i = 0;

        sForcesX[threadIdx.y * blockDim.x + threadIdx.x] = .0;
        sForcesY[threadIdx.y * blockDim.x + threadIdx.x] = .0;

        if (forceParticle < number_particles
            && targetParticle < number_particles) {
            /*
             * Mapping section
             */

            while (i < n) {
                int a = (forceParticle < number_particles);
                int b = ((forceParticle + blockDim.x) < number_particles);

                particle_t *fp_1 = &particles[forceParticle], *fp_2 = &particles[forceParticle + blockDim.x];
                particle_t *tp = &particles[targetParticle];
                // printf("values : %.6f ; %.6f\n", tp->x_pos, tp->y_pos);

                double x_sep_1 = fp_1->x_pos - tp->x_pos, x_sep_2 = fp_2->x_pos - tp->x_pos;
                double y_sep_1 = fp_1->y_pos - tp->y_pos, y_sep_2 = fp_2->y_pos - tp->y_pos;

                double dist_sq_1 = MAX((x_sep_1 * x_sep_1) + (y_sep_1 * y_sep_1), 0.01);
                double dist_sq_2 = MAX((x_sep_2 * x_sep_2) + (y_sep_2 * y_sep_2), 0.01);

                double grav_base_1 = GRAV_CONSTANT * (fp_1->mass) * (tp->mass) / dist_sq_1;
                double grav_base_2 = GRAV_CONSTANT * (fp_2->mass) * (tp->mass) / dist_sq_2;

                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        a * (grav_base_1 * x_sep_1) + b * (grav_base_2 * x_sep_2);
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        a * (grav_base_1 * y_sep_1) + b * (grav_base_2 * y_sep_2);

                forceParticle += gridSize;
                i++;
            }
            __syncthreads();

            /*
             * Reduce section
             */
            if (blockSize >= 512) {
                if (threadIdx.x < 256) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 256];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 256];
                }
                __syncthreads();
            }

            if (blockSize >= 256) {
                if (threadIdx.x < 128) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 128];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 128];
                }
                __syncthreads();
            }

            if (blockSize >= 128) {
                if (threadIdx.x < 64) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 64];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 64];
                }
                __syncthreads();
            }

            unsigned int s = blockDim.x / 2;

            if (blockSize >= 512) s >>= 3;
            else if (blockSize >= 256) s >>= 2;
            else if (blockSize >= 128) s >>= 1;

            if (threadIdx.x < s) {
                if (blockSize >= 64) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 32];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 32];
                    s >>= 1;
                }

                if (blockSize >= 32) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                    s >>= 1;
                }

                if (blockSize >= 16) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                    s >>= 1;
                }

                if (blockSize >= 8) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                    s >>= 1;
                }

                if (blockSize >= 4) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                    s >>= 1;
                }

                if (blockSize >= 2) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                }
            }

            if (!threadIdx.x) {
                //printf("sForcesX[%d] corresponding to particle %d are %f\n", threadIdx.y * blockDim.x, targetParticle, sForcesX[threadIdx.y * blockDim.x]);
                gForcesX[targetParticle * gridWidth + blockIdx.x] = sForcesX[threadIdx.y * blockDim.x];
                gForcesY[targetParticle * gridWidth + blockIdx.x] = sForcesY[threadIdx.y * blockDim.x];
            }
        }
    }

    // TODO : Having this in a separate method for this
    //  might lead to a small performance loss
    void call_kernel(int block_width, particle_t *particles, double *gForcesX,
                     double *gForcesY, const unsigned int number_particles,
                     const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block) {

        switch (block_width) {
            case 1024:
                calculate_forces<1024><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 512:
                calculate_forces<512><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 256:
                calculate_forces<256><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces<128><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces<64><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces<32><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces<16><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces<8><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces<4><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces<2><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces<1><<<grid, block>>>(particles, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    void cuda_nbody_all_pairs::calculate_forces() {
        uint size = number_particles * sizeof(particle_t);

        cudaMemcpy(gpu_particles, particles, size, cudaMemcpyHostToDevice);

        dim3 grid(gridWidth, gridHeight);
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
        call_kernel(BLOCK_WIDTH, gpu_particles, dForcesX, dForcesY, number_particles, gridWidth, n,grid, block);
        //::cadlabs::calculate_forces<256><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth, n);
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

    void cuda_nbody_all_pairs::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }


} // namespace
