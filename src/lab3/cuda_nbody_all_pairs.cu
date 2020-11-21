/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include <stdio.h>

static constexpr int BLOCK_WIDTH  = 256;
static constexpr int BLOCK_HEIGHT = 2;
static constexpr int n = 2;

static constexpr int thread_block_size = 512;
static constexpr int n_stream_width = 5, n_stream_height = 5;

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
        cudaMalloc(&gpu_particles, number_particles*sizeof(particle_t));
        gridWidth  = number_particles / (BLOCK_WIDTH * 2 * n) + (number_particles % (BLOCK_WIDTH * 2 * n) != 0);
        gridHeight = number_particles / (BLOCK_HEIGHT) + (number_particles % (BLOCK_HEIGHT) != 0);

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
        free(hForcesX);
        free(hForcesY);
        cudaFree(dForcesX);
        cudaFree(dForcesY);
    }

    template<unsigned int blockSize>
    __global__ void calculate_forces_two_cycles_parallel(
            particle_t *particles, double *gForcesX,
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

    template<unsigned int blockSize>
    __global__ void calculate_forces_two_cycles_parallel_soa(
            const double * __restrict__ x_pos, const double * __restrict__ y_pos,
            const double * __restrict__ mass,
            double * __restrict__ gForcesX, double * __restrict__ gForcesY,
            const unsigned int number_particles,
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

                // particle_t *fp_1 = &particles[forceParticle], *fp_2 = &particles[forceParticle + blockDim.x];
                // particle_t *tp = &particles[targetParticle];
                // printf("values : %.6f ; %.6f\n", tp->x_pos, tp->y_pos);

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

#ifdef SOA
    // TODO : Having this in a separate method for this
    //  might lead to a small performance loss
    static void call_kernel_soa(
            int block_width,
            const double * x_pos, const double * y_pos,
            const double * mass,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block) {

        switch (block_width) {
            case 1024:
                calculate_forces_two_cycles_parallel_soa<1024><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                            number_particles, gridWidth, n);
                break;
            case 512:
                calculate_forces_two_cycles_parallel_soa<512><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                           number_particles, gridWidth, n);
                break;
            case 256:
                calculate_forces_two_cycles_parallel_soa<256><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                           number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces_two_cycles_parallel_soa<128><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                           number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel_soa<64><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                          number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel_soa<32><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                          number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel_soa<16><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                          number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel_soa<8><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                         number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<4><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                         number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<2><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                         number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<1><<<grid, block>>>(x_pos, y_pos, mass, gForcesX, gForcesY,
                                                                         number_particles, gridWidth, n);
                break;
        }
    }

#else
    // TODO : Having this in a separate method for this
    //  might lead to a small performance loss
    static void call_kernel_aos(
            int block_width, particle_t *particles, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block) {

        switch (block_width) {
            case 1024:
                calculate_forces_two_cycles_parallel<1024><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                            number_particles, gridWidth, n);
                break;
            case 512:
                calculate_forces_two_cycles_parallel<512><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                           number_particles, gridWidth, n);
                break;
            case 256:
                calculate_forces_two_cycles_parallel<256><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                           number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces_two_cycles_parallel<128><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                           number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel<64><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                          number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel<32><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                          number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel<16><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                          number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel<8><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                         number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<4><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                         number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<2><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                         number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<1><<<grid, block>>>(particles, gForcesX, gForcesY,
                                                                         number_particles, gridWidth, n);
                break;
        }
    }
#endif

#ifdef SOA
    void cuda_nbody_all_pairs::calculate_forces() {
        uint count = number_particles * sizeof(double);
        cudaMemcpy(gpu_particles_soa.x_pos, particles_soa.x_pos, count, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_particles_soa.y_pos, particles_soa.y_pos, count, cudaMemcpyHostToDevice);
        dim3 grid(gridWidth, gridHeight);
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
        call_kernel_soa(BLOCK_WIDTH, gpu_particles_soa.x_pos, gpu_particles_soa.y_pos, gpu_particles_soa.mass,
                        dForcesX, dForcesY, number_particles, gridWidth, n, grid, block);
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
    void cuda_nbody_all_pairs::calculate_forces() {cudaStream_t streams [n_stream_height * n_stream_width];

        for (auto & stream : streams)
            cudaStreamCreate(&stream);

        int stream_size_x = number_particles / n_stream_width +
                            (number_particles % n_stream_width != 0);
        int stream_size_y = number_particles / n_stream_height +
                            (number_particles % n_stream_height!= 0);

        for (int i=0; i<n_stream_height; i++) {
            int pos_i = i * n_stream_width;
            for (int j = 0; j < n_stream_width; j++) {
                int index = pos_i + j;
                int stream_offset_x = j * stream_size_x, stream_offset_y = i * stream_size_y;
                dim3 temp_grid(stream_size_x / gridWidth + (stream_size_x % gridWidth != 0),
                               stream_size_y / gridHeight + (stream_size_y % gridHeight != 0));
                dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
                // TODO Isto provavelmente está mal
                cudaMemcpyAsync(&gpu_particles[stream_offset_x],
                                &particles[stream_offset_x],
                                stream_size_x * stream_size_y * sizeof(particle_t),
                                cudaMemcpyHostToDevice, streams[index]);

                // TODO Mudar isto de modo a chamar o método do switch
                calculate_forces_two_cycles_parallel<256><<<temp_grid, block, 0, streams[index]>>>(
                        &gpu_particles[stream_offset_x],
                        &dForcesX[stream_offset_x + stream_offset_y * gridWidth],
                        &dForcesY[stream_offset_x + stream_offset_y * gridWidth],
                        stream_size_x * stream_size_y,
                        stream_size_x / gridWidth + (stream_size_x % gridWidth != 0), n);

                cudaMemcpyAsync(&hForcesX[stream_offset_x + stream_offset_y * gridWidth],
                                &dForcesX[stream_offset_x + stream_offset_y * gridWidth],
                                stream_size_x * stream_size_y * gridWidth * sizeof(double),
                                cudaMemcpyDeviceToHost, streams[index]);
                cudaMemcpyAsync(&hForcesY[stream_offset_x + stream_offset_y * gridWidth],
                                &dForcesY[stream_offset_x + stream_offset_y * gridWidth],
                                stream_size_x * stream_size_y * gridWidth * sizeof(double),
                                cudaMemcpyDeviceToHost, streams[index]);
            }
        }

        for (auto & stream : streams)
            cudaStreamSynchronize(stream);

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
