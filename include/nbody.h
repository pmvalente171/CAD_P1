#ifndef NBODY_H
#define NBODY_H

#include <stdio.h>

enum class universe_t;


/*
  This structure holds information for a single particle,
  including position, velocity, and mass.
*/
typedef struct particle {
  double x_pos, y_pos;		/* position of the particle */
  double x_vel, y_vel;		/* velocity of the particle */
  double x_force, y_force;	/* gravitational forces that apply against this particle */
  double mass;			/* mass of the particle */
  struct node* node; 		/* only used for the barnes-hut algorithm */
} particle_t;


/* Only used in the barnes-Hut algorithm */
typedef struct node {
  struct node *parent;
  struct node *children;
  particle_t *particle;
  int n_particles; //number of particles in this node and its sub-nodes
  double mass; // mass of the node (ie. sum of its particles mass)
  double x_center, y_center; // center of the mass
  int depth;
  int owner;
  double x_min, x_max;
  double y_min, y_max;
} node_t;



/* used for debugging the display of the Barnes-Hut application */
#define DRAW_BOXES 1

#define DISPLAY_SIZE       512      /* pixel size of display window */
#define SCALE               0.03    /* sets the magnification at the origin */
                                    /* smaller #'s zoom in */
#define XMIN (-1/SCALE)
#define XMAX (1/SCALE)
#define YMIN (-1/SCALE)
#define YMAX (1/SCALE)

#define DISPLAY_RANGE       20      /* display range of fish space */
#define STEPS_PER_DISPLAY   10      /* time steps between display of fish */
#define GRAV_CONSTANT       0.01    /* proportionality constant of
                                       gravitational interaction */

#define POS_TO_SCREEN(pos)   ((int) ((pos/SCALE + DISPLAY_SIZE)/2))

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))  /* utility function */
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))  /* utility function */


#include "nbody_universe.h"

class nbody {

public:
    nbody(int number_particles, float t_final, universe_t universe);

    virtual ~nbody();

    void run_simulation();

    void print_all_particles(FILE* f);


    const int number_particles;      /* number of particles */
    const float T_FINAL;     /* simulation end time */
    const universe_t universe;


protected:

    virtual void all_move_particles(double step);

    void draw_all_particles();
    void compute_force(particle_t*p, double x_pos, double y_pos, double mass);
    void move_particle(particle_t*p, double step);
    void all_init_particles();

    particle_t* particles;

};

#endif
