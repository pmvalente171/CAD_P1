/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef DISPLAY
#include "ui.h"
#endif


#include "nbody.h"
#include "nbody_tools.h"

FILE* f_out=NULL;




double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;



nbody::nbody(int number_particles, float t_final, universe_t universe) :
        number_particles (number_particles),
        T_FINAL (t_final),
        universe (universe),
        particles (static_cast<particle_t *>(malloc(sizeof(particle_t) * number_particles)))  {

#ifdef DISPLAY
    /* Open an X window to display the particles */
    simple_init (100,100, DISPLAY_SIZE, DISPLAY_SIZE);
#endif

    all_init_particles();
}

nbody::~nbody() {
#ifdef DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();

    printf("Hit return to close the window.");

    getchar();
    /* Close the X window used to display the particles */
    XCloseDisplay(theDisplay);
#endif
}

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void nbody::compute_force(particle_t*p, double x_pos, double y_pos, double mass) {
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT*(p->mass)*(mass)/dist_sq;

  p->x_force += grav_base*x_sep;
  p->y_force += grav_base*y_sep;
}

/* compute the new position/velocity */
void nbody::move_particle(particle_t*p, double step) {

  p->x_pos += (p->x_vel)*step;
  p->y_pos += (p->y_vel)*step;
  double x_acc = p->x_force/p->mass;
  double y_acc = p->y_force/p->mass;
  p->x_vel += x_acc*step;
  p->y_vel += y_acc*step;

  /* compute statistics */
  double cur_acc = (x_acc*x_acc + y_acc*y_acc);
  cur_acc = sqrt(cur_acc);
  double speed_sq = (p->x_vel)*(p->x_vel) + (p->y_vel)*(p->y_vel);
  double cur_speed = sqrt(speed_sq);

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);
}


/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void nbody::all_move_particles(double step)
{
  /* First calculate force for particles. */
  int i;
  for(i=0; i<number_particles; i++) {
    int j;
    particles[i].x_force = 0;
    particles[i].y_force = 0;
    for(j=0; j<number_particles; j++) {
      particle_t*p = &particles[j];
      /* compute the force of particle j on particle i */
      compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
    }
  }

  /* then move all particles and return statistics */
  for(i=0; i<number_particles; i++) {
    move_particle(&particles[i], step);
  }
}


/* display all the particles */
void nbody::draw_all_particles() {
#ifdef DISPLAY
  int i;
  for(i=0; i<number_particles; i++) {
    int x = POS_TO_SCREEN(particles[i].x_pos);
    int y = POS_TO_SCREEN(particles[i].y_pos);
    draw_point (x,y);
  }
#endif
}

/*
  Place particles in their initial positions.
*/


void nbody::all_init_particles() {

    if (universe == universe_t::ORIGINAL) {
        printf("Universe: original\n");
        original(number_particles, particles);
    } else if (universe == universe_t::DISC) {
        printf("Universe: disc\n");
        rotating_disc(number_particles, particles);
    } else if (universe == universe_t::SPHERE) {
        printf("Universe: sphere\n");
        sphere(number_particles, particles);
    }
}

void nbody::print_all_particles(FILE* f) {
  int i;
  for(i=0; i<number_particles; i++) {
    particle_t*p = &particles[i];
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
}

void nbody::run_simulation() {
  double t = 0.0, dt = 0.01;
  while (t < T_FINAL && number_particles>0) {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles(dt);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    dt = 0.1*max_speed/max_acc;

    /* Plot the movement of the particle */
#if DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();
#endif
  }
}


