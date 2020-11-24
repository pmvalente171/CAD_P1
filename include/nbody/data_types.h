#ifndef CADLABS_GPU_DATA_TYPES_H
#define CADLABS_GPU_DATA_TYPES_H

// #define SOA
#define ATOMIC

#ifdef ATOMIC
typedef float force;
#else
typedef double force;
#endif

namespace data_types{

    struct node_t;

/*
 * This structure holds information for a single particle,
 * including position, velocity, and mass.
 */
    struct particle_t {
        double x_pos, y_pos;        /* position of the particle */
        double x_vel, y_vel;        /* velocity of the particle */
        force x_force, y_force;     /* gravitational forces that apply against this particle */
        double mass;                /* mass of the particle */
        node_t *node;               /* only used for the barnes-hut algorithm */
    };

    struct particle_soa {
        double *x_pos, *y_pos;
        double *x_vel, *y_vel;
        force *x_force, *y_force;
        double *mass;
    };

/*
 * Only used in the barnes-Hut algorithm
 */
    struct node_t {
        node_t *parent;
        node_t *children;
        particle_t *particle;
        int n_particles;            //number of particle_soa in this node and its sub-nodes
        double mass;                // mass of the node (ie. sum of its particle_soa mass)
        double x_center, y_center;  // center of the mass
        int depth;
        int owner;
        double x_min, x_max;
        double y_min, y_max;
    };

}

#endif //CADLABS_GPU_DATA_TYPES_H
