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
                const universe_t universe,
                const unsigned number_of_threads);



    protected:
        virtual void all_move_particles(double step);

        const unsigned number_of_threads;
    };
}

#endif //CADLABS_GPU_PAR_NBODY_ALL_PAIRS_H
