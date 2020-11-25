#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include <stdio.h>


namespace cadlabs {

    cuda_nbody_all_pairs::cuda_nbody_all_pairs(
            const int number_particles,
            const float t_final,
            const unsigned n,
            const universe_t universe,
            const unsigned universe_seed,
            const string file_name,
            const int blockWidth,
            const int blockHeight,
            const int n_streams) :
            nbody(number_particles, t_final, universe, universe_seed, file_name),
            blockWidth(blockWidth), n(n), numStreams(n_streams), blockHeight(blockHeight) {


#ifdef SOA
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
        gridWidth  = number_particles / (blockWidth * 2 * n) + (number_particles % (blockWidth * 2 * n) != 0);
        gridHeight = number_particles / (blockHeight) + (number_particles % (blockHeight) != 0);

        cudaMallocHost(&hForcesX, number_particles * gridWidth * sizeof(double));
        cudaMallocHost(&hForcesY, number_particles * gridWidth * sizeof(double));

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
        cudaFreeHost(hForcesX);
        cudaFreeHost(hForcesY);
        cudaFree(dForcesX);
        cudaFree(dForcesY);
    }


    //STREAM IMPLEMENTATION WITH ARRAYS OF STRUCTURES
    template<unsigned int blockSize, unsigned int blockHeight>
    __global__
    void calculate_forces_two_cycles_parallel(
            particle_t * __restrict__ particles, const unsigned int targetOffset,
            double * __restrict__ gForcesX, double * __restrict__ gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n) {

        __shared__ double sForcesX[blockHeight * blockSize];
        __shared__ double sForcesY[blockHeight * blockSize];

        unsigned int forceParticle  = blockIdx.x * 2 * blockDim.x + threadIdx.x;
        unsigned int targetParticle = blockIdx.y * blockDim.y + threadIdx.y + targetOffset;
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
                gForcesX[targetParticle * gridWidth + blockIdx.x] = sForcesX[threadIdx.y * blockDim.x];
                gForcesY[targetParticle * gridWidth + blockIdx.x] = sForcesY[threadIdx.y * blockDim.x];
            }
        }
    }

    template<unsigned int blockSize, unsigned int blockHeight>
    __global__ void calculate_forces_two_cycles_parallel_soa(
            const double * __restrict__ x_pos, const double * __restrict__ y_pos,
            const double * __restrict__ mass, const unsigned int target_offset,
            double * __restrict__ gForcesX, double * __restrict__ gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n) {

        __shared__ double sForcesX[blockHeight * blockSize];
        __shared__ double sForcesY[blockHeight * blockSize];

        unsigned int forceParticle  = blockIdx.x * 2 * blockDim.x + threadIdx.x;
        unsigned int targetParticle = blockIdx.y * blockDim.y + threadIdx.y + target_offset;
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
                gForcesX[targetParticle * gridWidth + blockIdx.x] = sForcesX[threadIdx.y * blockDim.x];
                gForcesY[targetParticle * gridWidth + blockIdx.x] = sForcesY[threadIdx.y * blockDim.x];
            }
        }
    }

#ifdef SOA
    static void call_kernel_1(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 1024:
                calculate_forces_two_cycles_parallel_soa<1, 1024><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 512:
                calculate_forces_two_cycles_parallel_soa<1, 512><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 256:
                calculate_forces_two_cycles_parallel_soa<1, 256><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces_two_cycles_parallel_soa<1, 128><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel_soa<1, 64><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel_soa<1, 32><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel_soa<1, 16><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel_soa<1, 8><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<1, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<1, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<1, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_2(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 512:
                calculate_forces_two_cycles_parallel_soa<2, 512><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 256:
                calculate_forces_two_cycles_parallel_soa<2, 256><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces_two_cycles_parallel_soa<2, 128><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel_soa<2, 64><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel_soa<2, 32><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel_soa<2, 16><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel_soa<2, 8><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<2, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<2, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<2, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_3(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 256:
                calculate_forces_two_cycles_parallel_soa<4, 256><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces_two_cycles_parallel_soa<4, 128><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel_soa<4, 64><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel_soa<4, 32><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel_soa<2, 16><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel_soa<4, 8><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<4, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<4, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<4, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_4(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 128:
                calculate_forces_two_cycles_parallel_soa<8, 128><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel_soa<8, 64><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel_soa<8, 32><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel_soa<8, 16><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel_soa<8, 8><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<8, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<8, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<8, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }


    static void call_kernel_5(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 64:
                calculate_forces_two_cycles_parallel_soa<16, 64><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel_soa<16, 32><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel_soa<16, 16><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel_soa<16, 8><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<16, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<16, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<16, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_6(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 32:
                calculate_forces_two_cycles_parallel_soa<32, 32><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel_soa<32, 16><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel_soa<32, 8><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<32, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<32, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<32, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_7(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 16:
                calculate_forces_two_cycles_parallel_soa<64, 16><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel_soa<64, 8><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<64, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<64, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<64, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_8(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 8:
                calculate_forces_two_cycles_parallel_soa<128, 8><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel_soa<128, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<128, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<128, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_9(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 4:
                calculate_forces_two_cycles_parallel_soa<256, 4><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel_soa<256, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<256, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_10(
            int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 2:
                calculate_forces_two_cycles_parallel_soa<512, 2><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel_soa<512, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
        }
    }

    // Having this in a separate method for this
    // might lead to a small performance loss
    static void call_kernel_soa(
            int block_width, int block_height,
            const double * x_pos, const double * y_pos,
            const double * mass, const int target_offset,
            double * gForcesX, double * gForcesY,
            const unsigned int number_particles,
            const unsigned int gridWidth,
            const unsigned int n, dim3 grid,
            dim3 block, cudaStream_t stream) {

        switch (block_width) {
            case 1024:
                calculate_forces_two_cycles_parallel_soa<1024, 1><<<grid, block, 0, stream>>>(
                        x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                        number_particles, gridWidth, n);
                break;
            case 512:
                call_kernel_10(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                               number_particles, gridWidth, n, grid, block, stream);
                break;
            case 256:
                call_kernel_9(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
            case 128:
                call_kernel_8(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
            case 64:
                call_kernel_7(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
            case 32:
                call_kernel_6(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
            case 16:
                call_kernel_5(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
            case 8:
                call_kernel_4(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
            case 4:
                call_kernel_3(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
            case 2:
                call_kernel_2(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
            case 1:
                call_kernel_1(block_height, x_pos, y_pos, mass, target_offset, gForcesX, gForcesY,
                              number_particles, gridWidth, n, grid, block, stream);
                break;
        }
    }
#else
    static void call_kernel_1(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 1024:
                calculate_forces_two_cycles_parallel<1, 1024><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 512:
                calculate_forces_two_cycles_parallel<1, 512><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 256:
                calculate_forces_two_cycles_parallel<1, 256><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces_two_cycles_parallel<1, 128><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel<1, 64><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel<1, 32><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel<1, 16><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel<1, 8><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<1, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<1, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<1, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_2(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 512:
                calculate_forces_two_cycles_parallel<2, 512><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 256:
                calculate_forces_two_cycles_parallel<2, 256><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces_two_cycles_parallel<2, 128><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel<2, 64><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel<2, 32><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel<2, 16><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel<2, 8><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<2, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<2, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<2, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_3(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 256:
                calculate_forces_two_cycles_parallel<4, 256><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 128:
                calculate_forces_two_cycles_parallel<4, 128><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel<4, 64><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel<4, 32><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel<2, 16><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel<4, 8><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<4, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<4, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<4, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_4(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 128:
                calculate_forces_two_cycles_parallel<8, 128><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 64:
                calculate_forces_two_cycles_parallel<8, 64><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel<8, 32><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel<8, 16><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel<8, 8><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<8, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<8, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<8, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }


    static void call_kernel_5(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 64:
                calculate_forces_two_cycles_parallel<16, 64><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 32:
                calculate_forces_two_cycles_parallel<16, 32><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel<16, 16><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel<16, 8><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<16, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<16, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<16, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_6(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 32:
                calculate_forces_two_cycles_parallel<32, 32><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 16:
                calculate_forces_two_cycles_parallel<32, 16><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel<32, 8><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<32, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<32, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<32, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_7(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 16:
                calculate_forces_two_cycles_parallel<64, 16><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 8:
                calculate_forces_two_cycles_parallel<64, 8><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<64, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<64, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<64, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_8(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 8:
                calculate_forces_two_cycles_parallel<128, 8><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 4:
                calculate_forces_two_cycles_parallel<128, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<128, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<128, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_9(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 4:
                calculate_forces_two_cycles_parallel<256, 4><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 2:
                calculate_forces_two_cycles_parallel<256, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<256, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    static void call_kernel_10(
            int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_height) {
            case 2:
                calculate_forces_two_cycles_parallel<512, 2><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 1:
                calculate_forces_two_cycles_parallel<512, 1><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
        }
    }

    // Having this in a separate method for this
    // might lead to a small performance loss
    static void call_kernel_aos(
            int block_width, int block_height, particle_t *particles, int targetOffset, double *gForcesX,
            double *gForcesY, const unsigned int number_particles,
            const unsigned int gridWidth, const unsigned int n, dim3 grid, dim3 block, cudaStream_t stream) {

        switch (block_width) {
            case 1024:
                calculate_forces_two_cycles_parallel<1, 1024><<<grid, block, 0, stream>>>(
                        particles, targetOffset, gForcesX, gForcesY, number_particles, gridWidth, n);
                break;
            case 512:
                call_kernel_10(block_height, particles, targetOffset,gForcesX, gForcesY,
                                  number_particles, gridWidth, n, grid, block, stream);
                break;
            case 256:
                call_kernel_9(block_height, particles, targetOffset,gForcesX, gForcesY,
                                  number_particles, gridWidth, n, grid, block, stream);
                break;
            case 128:
                call_kernel_8(block_height, particles, targetOffset,gForcesX, gForcesY,
                                  number_particles, gridWidth, n, grid, block, stream);
                break;
            case 64:
                call_kernel_7(block_height, particles, targetOffset,gForcesX, gForcesY,
                                  number_particles, gridWidth, n, grid, block, stream);
                break;
            case 32:
                call_kernel_6(block_height, particles, targetOffset,gForcesX, gForcesY,
                                number_particles, gridWidth, n, grid, block, stream);
                break;
            case 16:
                call_kernel_5(block_height, particles, targetOffset,gForcesX, gForcesY,
                                number_particles, gridWidth, n, grid, block, stream);
                break;
            case 8:
                call_kernel_4(block_height, particles, targetOffset,gForcesX, gForcesY,
                                number_particles, gridWidth, n, grid, block, stream);
                break;
            case 4:
                call_kernel_3(block_height, particles, targetOffset,gForcesX, gForcesY,
                               number_particles, gridWidth, n, grid, block, stream);
                break;
            case 2:
                call_kernel_2(block_height, particles, targetOffset,gForcesX, gForcesY,
                               number_particles, gridWidth, n, grid, block, stream);
                break;
            case 1:
                call_kernel_1(block_height, particles, targetOffset,gForcesX, gForcesY,
                               number_particles, gridWidth, n, grid, block, stream);
                break;
        }
    }
#endif

#ifdef SOA
    void cuda_nbody_all_pairs::calculate_forces() {
        cudaStream_t streams[numStreams];
        cudaEvent_t events[numStreams];
        uint size = number_particles * sizeof(double);
        dim3 block(blockWidth, blockHeight);

        cudaMemcpy(gpu_particles_soa.x_pos, particles_soa.x_pos, size, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_particles_soa.y_pos, particles_soa.y_pos, size, cudaMemcpyHostToDevice);

        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&events[i]);

            unsigned int partialHeight = (gridHeight / numStreams) +
                    (i == numStreams - 1 && (gridHeight % numStreams)) *
                    gridHeight % numStreams;
            int temp = partialHeight * blockHeight;
            int targetOffset = (int)(i * blockHeight * (gridHeight / numStreams));
            dim3 partialGrid(gridWidth, partialHeight);

            call_kernel_soa(
                    blockWidth, blockHeight, gpu_particles_soa.x_pos, gpu_particles_soa.y_pos,
                    gpu_particles_soa.mass, targetOffset, dForcesX, dForcesY,
                    number_particles, gridWidth, n, partialGrid, block, streams[i]);

            cudaMemcpyAsync(&hForcesX[targetOffset], &dForcesX[targetOffset],
                            temp * sizeof(double),
                            cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(&hForcesY[targetOffset], &dForcesY[targetOffset],
                            temp * sizeof(double),
                            cudaMemcpyDeviceToHost, streams[i]);
            cudaEventRecord(events[i], streams[i]);
        }

        for (int s=0; s<numStreams; s++) {
            cudaEventSynchronize(events[s]);

            unsigned int padding = (gridHeight / numStreams) +
                    (s == numStreams - 1 && (gridHeight % numStreams)) *
                    (gridHeight % numStreams);

            padding *= blockHeight;
            unsigned int offset = (int)(s * blockHeight * (gridHeight / numStreams));
            for (unsigned int i = offset; i < offset + padding; i++) {
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
    }
#else
    void cuda_nbody_all_pairs::calculate_forces() {
        cudaStream_t streams[numStreams];
        cudaEvent_t events[numStreams];
        uint size = number_particles * sizeof(particle_t);
        dim3 block(blockWidth, blockHeight);

        cudaMemcpy(gpu_particles, particles, size, cudaMemcpyHostToDevice);

        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&events[i]);

            unsigned int partialHeight = (gridHeight / numStreams) +
                    (i == numStreams - 1 && (gridHeight % numStreams)) *
                    gridHeight % numStreams;
            int temp = partialHeight * blockHeight;
            int targetOffset = (int)(i * blockHeight * (gridHeight / numStreams));
            dim3 partialGrid(gridWidth, partialHeight);

            call_kernel_aos(blockWidth, blockHeight,  gpu_particles, targetOffset,
                            dForcesX, dForcesY, number_particles, gridWidth,
                            n, partialGrid, block, streams[i]);

            cudaMemcpyAsync(&hForcesX[targetOffset], &dForcesX[targetOffset],
                            temp * sizeof(double),
                            cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(&hForcesY[targetOffset], &dForcesY[targetOffset],
                            temp * sizeof(double),
                            cudaMemcpyDeviceToHost, streams[i]);
            cudaEventRecord(events[i], streams[i]);
        }

        for (int s=0; s<numStreams; s++) {
            cudaEventSynchronize(events[s]);

            unsigned int padding = (gridHeight / numStreams) +
                    (s == numStreams - 1 && (gridHeight % numStreams)) *
                    (gridHeight % numStreams);

            padding *= blockHeight;
            unsigned int offset = (int)(s * blockHeight * (gridHeight / numStreams));
            for (unsigned int i = offset; i < offset + padding; i++) {
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
    }
#endif

    void cuda_nbody_all_pairs::move_all_particles(double step) {
        nbody::move_all_particles(step);
    }

    void cuda_nbody_all_pairs::print_all_particles(std::ostream &out) {
        nbody::print_all_particles(out);
    }

} // namespace
