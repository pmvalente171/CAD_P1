//
// Created by Hervé Paulino on 29/09/2020.
//

#ifndef CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H
#define CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H

#include "nbody.h"

namespace cadlabs {

    class cuda_nbody_all_pairs : public nbody {

        int gridWidth;   //(number_particles / BLOCK_WIDTH)
        int gridHeight;  //(number_particles / BLOCK_HEIGHT)

        /*
         * Matrix (number_particles x gridWidth)
         * Holds, for each particle an array with the forces to be applied to it.
         * The forces in these arrays correspond to the effects of the particles on each others
         * Each element of the arrays represent a reduction of the
         */
        double *hForcesX; //(number_particles x gridWidth)
        double *hForcesY;

        double *dForcesX;
        double *dForcesY;

    public:
        cuda_nbody_all_pairs(
                const int number_particles,
                const float t_final,
                const unsigned number_of_threads,
                const universe_t universe,
                const unsigned universe_seed = 0,
                const string file_name = "");

        ~cuda_nbody_all_pairs();

        void print_all_particles(std::ostream &out);

    protected:
        void calculate_forces();

        void all_init_particles();

        void move_all_particles(double step);

        particle_t *gpu_particles;

        const unsigned number_blocks;

    };
}

#endif //CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H
