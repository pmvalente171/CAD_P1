/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include <stdio.h>

static constexpr int BLOCK_WIDTH  = 256;
static constexpr int BLOCK_HEIGHT = 1;

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

    template<unsigned int blockSize>
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

            //if (!targetParticle)
              //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particles %d and %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, forceParticle + blockDim.x, targetParticle);

            __syncthreads();

            double ret = 0;
            for (int i = 0; i < number_particles; i++)
                ret += sForcesX[i];

            /*
             * Reduce section
             */
            //unsigned int s = 512;

            if (blockSize == 1024) {
                //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                 //   printf("Block size of 1024\n");
                if (threadIdx.x < 512) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 512];
                    //if (!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 512, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 512, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 512];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                __syncthreads();
            }
            //s >>= 1;

            if (blockSize >= 512) {
                //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                  //  printf("Block size of 512\n");
                if (threadIdx.x < 256) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 256];
                    //if(!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 256, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 256, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 256];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                __syncthreads();
            }
            //s >>= 1;

            if (blockSize >= 256) {
                //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                    //printf("Block size of 256\n");
                if (threadIdx.x < 128) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 128];
                    //if(!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 128, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 128, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 128];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                __syncthreads();
            }
            //s >>= 1;

            if (blockSize >= 128) {
                //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                  //  printf("Block size of 128\n");
                if (threadIdx.x < 64) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 64];
                    //if(!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 64, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 64, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 64];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                __syncthreads();
            }
            //s >>= 1;

            if (threadIdx.x < 32) {
                if (blockSize >= 64) {
                    //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                        //printf("Block size of 64\n");
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 32];
                    //if(!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 32, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 32, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 32];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                //s >>= 1;

                if (blockSize >= 32) {
                    //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                        //printf("Block size of 32\n");
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 16];
                    //if(!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 16, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 16, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 16];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                //s >>= 1;

                if (blockSize >= 16) {
                    //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                       // printf("Block size of 16\n");
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 8];
                    //if(!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 8, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 8, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 8];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                //s >>= 1;

                if (blockSize >= 8) {
                    if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                        //printf("Block size of 8\n");
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 4];
                    //if(!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 4, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 4, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 4];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                //s >>= 1;

                if (blockSize >= 4) {
                    //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                        //printf("Block size of 4\n");
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 2];
                    //if(!targetParticle)
                        //printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 2, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 2, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 2];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
                //s >>= 1;

                if (blockSize >= 2) {
                    //if (/*!blockIdx.x && !blockIdx.y &&*/ !threadIdx.x && !threadIdx.y)
                        //printf("Block size of 2\n");
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 1];
                    //if(!targetParticle)
                      //  printf("Adding sForcesX[%d] with sForcesX[%d], resulting in %f, corresponding to particles %d and %d's effect on %d\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + 1, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], forceParticle, forceParticle + 1, targetParticle);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 1];
                }
                //if (!threadIdx.x)
                  //  printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);
            }

            /*for(s = (blockDim.x)/2; s > 32 ; s>>=1) {
                //printf("S value: %d\n", s);
                if (threadIdx.x < s) {
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                }
                __syncthreads();
            }*/

            //printf("S value: %d\n", s);
            /*if (threadIdx.x < s) {
                //printf("S value: %d\n", s);
                sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesX[threadIdx.y * blockDim.x + threadIdx.x + 32];
                sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                        sForcesY[threadIdx.y * blockDim.x + threadIdx.x + 32];
                s >>= 1;

                //printf("S value: %d\n", s);
                if (s != 16)
                    printf ("In thread(%d, %d) corresponding to %d's effect on %d the S value is %d", threadIdx.y, threadIdx.x, forceParticle, targetParticle, s);
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
            }*/

            if (!threadIdx.x) {
                if (ret != sForcesX[0])
                    printf("Ret is %f while sForcesX is %f for particle %d\n", ret, sForcesX[0], targetParticle);
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

        switch(thread_block_size) {
            case 1024:
                ::cadlabs::calculate_forces<1024><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 512:
                ::cadlabs::calculate_forces<512><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 256:
                ::cadlabs::calculate_forces<256><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 128:
                ::cadlabs::calculate_forces<128><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 64:
                ::cadlabs::calculate_forces<64><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 32:
                ::cadlabs::calculate_forces<32><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 16:
                ::cadlabs::calculate_forces<16><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 8:
                ::cadlabs::calculate_forces<8><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 4:
                ::cadlabs::calculate_forces<4><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 2:
                ::cadlabs::calculate_forces<2><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
            case 1:
                ::cadlabs::calculate_forces<1><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
                break;
        }
        ::cadlabs::calculate_forces<512><<<grid, block>>>(gpu_particles, dForcesX, dForcesY, number_particles, gridWidth);
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
