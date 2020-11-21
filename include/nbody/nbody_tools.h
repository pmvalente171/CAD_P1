#ifndef NBODY_TOOLS_H
#define NBODY_TOOLS_H
#include "nbody.h"
#include "nbody_universe.h"

namespace cadlabs {
/* draw recursively the content of a node */
    void draw_node(node_t *n);

/* print recursively the particle_soa of a node */
    void print_particles(FILE *f, node_t *n);

/* Initialize a node */
    void init_node(node_t *n, node_t *parent, double x_min, double x_max, double y_min, double y_max);

/* Compute the position of a particle in a node and return
 * the quadrant in which it should be placed
 */
    int get_quadrant(particle_t *particle, node_t *node);

/* inserts a particle in a node (or one of its children)  */
    void insert_particle(particle_t *particle, node_t *node);


    void init_alloc(int nb_blocks);

    void free_node(node_t *n);

    node_t *alloc_node();

    void free_root(node_t *root);

}
#endif	/* NBODY_TOOLS_H */
