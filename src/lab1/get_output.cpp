//
// Created by pm_valente on 07/11/20.
//

#include "get_output.h"

void get_output::save_values_by_iteration(particle_t *particles, int nb_of_particles)
{
    // verify if the file is not open
    if(!outfile.is_open())
        return;

    particle_t temp_particle{};
    // Add the new positions to the file
    for (int i=0; i<nb_of_particles; i++) {
        // particle
        temp_particle = particles[i]; // TODO: Not sure i this works

        // Create the string
        char buffer [30];
        sprintf(buffer, "particle %d : %.3f %.3f ;\n", i,
                temp_particle.x_pos, temp_particle.y_pos);

        outfile << buffer;
    }
}

get_output::get_output(const string& file_name) {
    outfile.open(file_name);
}

get_output::~get_output() {
    outfile.close();
}
