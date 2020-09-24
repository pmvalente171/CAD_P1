/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>

#ifdef DISPLAY
#include "ui.h"
#endif

#include<omp.h>


#include "nbody.h"
#include "nbody_tools.h"

int number_of_threads;

FILE* f_out=NULL;

int nparticles=10;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
particle_t*particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;


#ifdef DISPLAY
Display *theDisplay;  /* These three variables are required to open the */
GC theGC;             /* particle plotting window.  They are externally */
Window theMain;       /* declared in ui.h but are also required here.   */
#endif

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force(particle_t*p, double x_pos, double y_pos, double mass) {
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
void move_particle(particle_t*p, double step) {

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
void all_move_particles(double step)
{
  /* First calculate force for particles. */
  int i;
#pragma omp parallel for num_threads(number_of_threads)
  for(i=0; i<nparticles; i++) {

    int j;
    particles[i].x_force = 0;
    particles[i].y_force = 0;
    for(j=0; j<nparticles; j++) {
      particle_t*p = &particles[j];
      /* compute the force of particle j on particle i */
      compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
    }
  }

  /* then move all particles and return statistics */
  for(i=0; i<nparticles; i++) {
    move_particle(&particles[i], step);
  }
}

/* display all the particles */
void draw_all_particles() {
#ifdef DISPLAY
  int i;
  for(i=0; i<nparticles; i++) {
    int x = POS_TO_SCREEN(particles[i].x_pos);
    int y = POS_TO_SCREEN(particles[i].y_pos);
    draw_point (x,y);
  }
#endif
}

void print_all_particles(FILE* f) {
  int i;
  for(i=0; i<nparticles; i++) {
    particle_t*p = &particles[i];
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
}

void run_simulation() {
  double t = 0.0, dt = 0.01;
  while (t < T_FINAL && nparticles>0) {
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




void init(int argc, char* argv[]) {
    int c;
    while ((c = getopt (argc-1, argv+1, "t:u:s:n:")) != -1)
        switch (c) {
            case 't':
                T_FINAL = atof(optarg);
                break;

            case 'n':
                number_of_threads = atoi(optarg);
                break;

            case 'u':
                universe = atoi(optarg);
                break;
            case 's':
                //         universe_seed = atoi(optarg);;
                break;

            default:
                fprintf (stderr, "%c option not supported\n", c);
                usage(argv[0]);
                exit(1);
        }

}

void usage(char* prog) {

    fprintf (stderr, "usage: %s number_particles [-t duration time] [-n number_of_threads] [-u universe] [-s seed]\n"
                     "\t-t --> number of end time (default 1.0)\n"
                     "\t-n --> number of threads (default number of hardware threads)\n"
                     "\t-u --> universe type [0 - line, 1 - sphere, 2 - rotating disc] (default 0)\n"
                     "\t-s --> seed for universe creation. Used in disc.", prog);
}