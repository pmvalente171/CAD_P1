//
// Created by pm_valente on 07/11/20.
//

#include <fstream>
#include <iostream>
#include <stdio.h>
#include "nbody/data_types.h"

using namespace std;
using namespace data_types;

#ifndef CADLABS_GPU_GET_OUTPUT_H
#define CADLABS_GPU_GET_OUTPUT_H

class get_output {
public:
    void save_values_by_iteration(particle_t *particles, int nb_of_particles);
    explicit get_output(const string& file_name);
    ~get_output();

private:
    ofstream outfile;
};


#endif //CADLABS_GPU_GET_OUTPUT_H
