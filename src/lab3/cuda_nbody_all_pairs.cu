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

        cudaMallocHost(&hForcesX, number_particles * gridWidth * sizeof(double));
        cudaMallocHost(&hForcesY, number_particles * gridWidth * sizeof(double));

        cudaMalloc(&dForcesX, number_particles * gridWidth * sizeof(double));
        cudaMalloc(&dForcesY, number_particles * gridWidth * sizeof(double));
    }

    cuda_nbody_all_pairs::~cuda_nbody_all_pairs() {
        cudaFree(gpu_particles);
        cudaFreeHost(hForcesX);
        cudaFreeHost(hForcesY);
        cudaFree(dForcesX);
        cudaFree(dForcesY);
    }

    template<unsigned int blockSize>
    __global__ void calculate_forces(particle_t *particles, const unsigned int targetOffset,
                                     double *gForcesX, double *gForcesY,
                                     const unsigned int number_particles,
                                     const unsigned int gridWidth, const unsigned int n) {

        __shared__ double sForcesX[BLOCK_HEIGHT * BLOCK_WIDTH];
        __shared__ double sForcesY[BLOCK_HEIGHT * BLOCK_WIDTH];

        unsigned int forceParticle  = blockIdx.x * 2 * blockDim.x + threadIdx.x;
        unsigned int targetParticle = blockIdx.y * blockDim.y + threadIdx.y + targetOffset;
        unsigned int gridSize = blockDim.x * 2 * gridDim.x, i = 0;

        //printf("Thread(%d, %d)\n", forceParticle, targetParticle);

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

            // printf("S value: %d\n", s);
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

    void cuda_nbody_all_pairs::calculate_forces() {
        uint size = number_particles * sizeof(particle_t);
        cudaMemcpy(gpu_particles, particles, size, cudaMemcpyHostToDevice);
        const uint numStreams = 2;
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);

        cudaStream_t streams[numStreams];
        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
            int offset = i * (gridHeight * BLOCK_HEIGHT / numStreams) * gridWidth;
            //printf("Offset: %d\n", offset);
            int partialHeight = (gridHeight / numStreams) + (i == numStreams - 1 && (gridHeight % numStreams));
            dim3 partialGrid(gridWidth, partialHeight);
            //printf("Starting from particle %d\n", i * BLOCK_HEIGHT * (gridHeight / numStreams));
            int targetOffset = i * BLOCK_HEIGHT * (gridHeight / numStreams);
            //printf("TargetOffset: %d\n", targetOffset);
            ::cadlabs::calculate_forces<256><<<partialGrid, block>>>(gpu_particles,
                                                                     targetOffset,
                                                                     dForcesX, dForcesY,
                                                                     number_particles, gridWidth, n);
            cudaMemcpyAsync(&hForcesX[offset], &dForcesX[offset], partialHeight * BLOCK_HEIGHT * gridWidth * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(&hForcesY[offset], &dForcesY[offset], partialHeight * BLOCK_HEIGHT * gridWidth * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
            //printf("blocks retreived: %d\n", partialHeight * BLOCK_HEIGHT * gridWidth);
            //printf("Stream %d received data from blocks %d to %d\n", i, offset, offset + partialHeight * BLOCK_HEIGHT * gridWidth);
        }
        /*for (int i = 0; i < numStreams; i++)
            cudaStreamSynchronize(streams[i]);*/
        cudaDeviceSynchronize();

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
        //printf("\n");
    }

    void cuda_nbody_all_pairs::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }


} // namespace
