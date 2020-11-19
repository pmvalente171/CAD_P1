/**
 * Hervé Paulino
 */

#include <nbody/cuda_nbody_all_pairs.h>
#include <omp.h>
#include <stdio.h>

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
        gridWidth  = number_particles / BLOCK_WIDTH + (number_particles % BLOCK_WIDTH != 0);
        gridHeight = number_particles / BLOCK_HEIGHT + (number_particles % BLOCK_HEIGHT != 0);
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
        sForcesX[threadIdx.y * blockDim.x + threadIdx.x] = .0;
        sForcesY[threadIdx.y * blockDim.x + threadIdx.x] = .0;
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
            sForcesX[threadIdx.y * blockDim.x + threadIdx.x] = grav_base * x_sep;
            sForcesY[threadIdx.y * blockDim.x + threadIdx.x] = grav_base * y_sep;

            //printf ("Thread(%d, %d) placed %f in SForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);

            __syncthreads();

            /*if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                printf("HERE\n");
                for (int i = 0; i < blockDim.x; i++)
                    printf("%f ", sForcesX[i]);
                printf("\n");
            }*/

            /*
             * Reduce section
             */
            for(unsigned int s = 1; s < blockDim.x; s *= 2) {
                if (!(threadIdx.x % (2 * s))) {
                    //printf("ThreadIdx.x %d\n", threadIdx.x);
                    sForcesX[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesX[threadIdx.y * blockDim.x + threadIdx.x + s];
                    // printf("Thread : X:%d; Y:%d; ;Write position : %d ; Read position : %d\n", threadIdx.x, threadIdx.y, threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + s);
                    //printf("sForcesX[%d] += sForces[%d]\n", threadIdx.y * blockDim.x + threadIdx.x, threadIdx.y * blockDim.x + threadIdx.x + s);
                    //printf("Result: sForcesX[%d] = %f\n", threadIdx.y * blockDim.x + threadIdx.x, sForcesX[threadIdx.y * blockDim.x + threadIdx.x]);
                    sForcesY[threadIdx.y * blockDim.x + threadIdx.x] +=
                            sForcesY[threadIdx.y * blockDim.x + threadIdx.x + s];
                }
                __syncthreads();
                // if (threadIdx.x == 0 && threadIdx.y == 0) printf("oof blockId : %d; %d ||| S : %d \n", blockIdx.x, blockIdx.y, s);
            }

            //printf ("Thread(%d, %d) placed %f in sForcesX[%d], corresponding to particle %d's effect on %d\n", threadIdx.x, threadIdx.y, sForcesX[threadIdx.y * blockDim.x + threadIdx.x], threadIdx.y * blockDim.x + threadIdx.x, forceParticle, targetParticle);

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
                //printf("index %d (i:%d, j:%d)\n", targetParticle + j, i, j);
                //printf("hForcesX[%d] = %f, for particle %d\n", i, hForcesX[targetParticle + j], targetParticle);
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
