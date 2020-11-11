//
// Created by pm_valente on 08/11/20.
//

#define SOA
#define AOS

#ifndef CADLABS_GPU_DATA_TYPES_H
#define CADLABS_GPU_DATA_TYPES_H

namespace data_types{

    struct node_t;

    /*
     * This structure holds information for a single particle,
     * including position, velocity, and mass.
    */
    struct particle_t {
        double x_pos, y_pos;        /* position of the particle */
        double x_vel, y_vel;        /* velocity of the particle */
        double x_force, y_force;    /* gravitational forces that apply against this particle */
        double mass;            /* mass of the particle */
        node_t *node;        /* only used for the barnes-hut algorithm */
    };


    // TODO: Add a bunch of
    //  templated functions
    //  to help out with this
    //  structs
    struct particles {
        float *x_pos, *y_pos;
        float *x_vel, *y_vel;
        float *x_force, *y_force;
        float *mass;
    };


    /* Only used in the barnes-Hut algorithm */
    struct node_t {
        node_t *parent;
        node_t *children;
        particle_t *particle;
        int n_particles; //number of particles in this node and its sub-nodes
        double mass; // mass of the node (ie. sum of its particles mass)
        double x_center, y_center; // center of the mass
        int depth;
        int owner;
        double x_min, x_max;
        double y_min, y_max;
    };

}

#endif //CADLABS_GPU_DATA_TYPES_H
