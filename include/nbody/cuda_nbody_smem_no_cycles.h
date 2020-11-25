//
// Created by Hervé Paulino on 29/09/2020.
//

#ifndef CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H
#define CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H

#include "nbody.h"

namespace cadlabs {

    class cuda_nbody_smem_no_cycles : public nbody {

        int gridWidth;   //(number_particles / blockWidth)
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
        cuda_nbody_smem_no_cycles(
                int number_particles,
                float t_final,
                unsigned n,
                universe_t universe,
                unsigned universe_seed = 0,
                string file_name = "",
                int blockWidth = 256,
                int blockHeight = 2,
                int n_streams = 2);

        ~cuda_nbody_smem_no_cycles() override;

        void print_all_particles(std::ostream &out);

    protected:
        void calculate_forces() override;

        void all_init_particles();

        void move_all_particles(double step) override;

        particle_t *gpu_particles;

        particle_soa gpu_particles_soa;

        const unsigned n;

        const unsigned blockWidth;

        const unsigned blockHeight;

        const unsigned numStreams;

    };
}

#endif //CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H
