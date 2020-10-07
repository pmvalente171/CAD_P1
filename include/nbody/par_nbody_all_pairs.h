//
// Created by Hervé Paulino on 29/09/2020.
//

#ifndef CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H
#define CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H

#include "nbody.h"

namespace cadlabs {

    class par_nbody_all_pairs : public nbody {

    public:
        par_nbody_all_pairs(
                const int number_particles,
                const float t_final,
                const unsigned number_of_threads,
                const universe_t universe,
                const unsigned universe_seed = 0);



    protected:
        void calculate_forces();

        const unsigned number_of_threads;
    };
}

#endif //CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H
