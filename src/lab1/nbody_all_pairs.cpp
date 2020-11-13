/*
 * nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
 *
 */

#include "nbody.h"

#include <math.h>
#include <stdlib.h>
#include <ostream>
#include "nbody/nbody_universe.h"

#define DEBUG

#ifdef DISPLAY
#include "nbody/ui.h"
#endif

namespace cadlabs {

    double sum_speed_sq = 0;
    double max_acc = 0;
    double max_speed = 0;


    nbody::nbody(const int number_particles, const float t_final, const universe_t universe, const unsigned universe_seed, const string file_name) :
            number_particles(number_particles),
            T_FINAL(t_final),
            universe(universe),
            universe_seed (universe_seed),
            particles(static_cast<particle_t *>(malloc(sizeof(particle_t) * number_particles))) {

#ifdef SOA
        particles_soa.x_pos = static_cast<double *>(malloc(sizeof(double) * number_particles));
        particles_soa.y_pos = static_cast<double *>(malloc(sizeof(double) * number_particles));

        particles_soa.x_vel = static_cast<double *>(malloc(sizeof(double) * number_particles));
        particles_soa.y_vel = static_cast<double *>(malloc(sizeof(double) * number_particles));

        particles_soa.x_force = static_cast<double *>(malloc(sizeof(double) * number_particles));
        particles_soa.y_force = static_cast<double *>(malloc(sizeof(double) * number_particles));

        particles_soa.mass = static_cast<double *>(malloc(sizeof(double) * number_particles));
#endif
        all_init_particles();

#ifdef DISPLAY
        /* Open an X window to display the particle_soa */
        simple_init (100,100, DISPLAY_SIZE, DISPLAY_SIZE);
#endif
#ifdef DEBUG
        debug = new get_output(file_name);
        // TODO: Maybe change this?
#endif
    }

    nbody::~nbody() {
#ifdef DISPLAY
        clear_display();
        draw_all_particles();
        flush_display();

        printf("Hit return to close the window.");

        getchar();
        /* Close the X window used to display the particle_soa */
        XCloseDisplay(theDisplay);
#endif
#ifdef DEBUG
        debug->~get_output();
#endif
        free(particles);
#ifdef SOA
        free(particles_soa.mass);

        free(particles_soa.x_pos);
        free(particles_soa.x_vel);
        free(particles_soa.x_force);

        free(particles_soa.y_pos);
        free(particles_soa.y_vel);
        free(particles_soa.y_force);
#endif
    }

    void nbody::reset() {
        all_init_particles();
    }

/*
 * compute the force that a
 * particle with position
 * (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
    void nbody::compute_force(particle_t *p, double x_pos, double y_pos, double mass) {
        double x_sep, y_sep, dist_sq, grav_base;

        x_sep = x_pos - p->x_pos;
        y_sep = y_pos - p->y_pos;
        dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

        /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
        grav_base = GRAV_CONSTANT * (p->mass) * (mass) / dist_sq;

        p->x_force += grav_base * x_sep;
        p->y_force += grav_base * y_sep;
    }

/*
 *  Like the method above this function calculates
 *  the force between two particles
 *  but does this for an SOA
 *  organization of the data
 */
    void nbody::compute_force(
            const double * const x_pos, const double * const y_pos,
            double * const x_force, double * const y_force, const double *const mass,
            double other_x_pos, double other_y_pos, double other_mass) {

        double x_sep, y_sep, dist_sq, grav_base;

        x_sep = other_x_pos - *x_pos;
        y_sep = other_y_pos - *y_pos;
        dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

        /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
        grav_base = GRAV_CONSTANT * (*mass) * (other_mass) / dist_sq;

        *x_force += grav_base * x_sep;
        *y_force += grav_base * y_sep;
    }

/*
 * compute the new position/velocity
 */
    void nbody::move_particle(particle_t *p, double step) {
        p->x_pos += (p->x_vel) * step;
        p->y_pos += (p->y_vel) * step;
        double x_acc = p->x_force / p->mass;
        double y_acc = p->y_force / p->mass;
        p->x_vel += x_acc * step;
        p->y_vel += y_acc * step;

        /* compute statistics */
        double cur_acc = (x_acc * x_acc + y_acc * y_acc);
        cur_acc = sqrt(cur_acc);
        double speed_sq = (p->x_vel) * (p->x_vel) + (p->y_vel) * (p->y_vel);
        double cur_speed = sqrt(speed_sq);

        sum_speed_sq += speed_sq;
        max_acc = MAX(max_acc, cur_acc);
        max_speed = MAX(max_speed, cur_speed);
    }

/*
 * This method is equivalent to the method
 * above but it is set up for a soa organization
 * of the data and it's used to compute the new
 * position/velocity
 */
    void nbody::move_particle(
            double * const x_pos, double * const y_pos,
            double * const x_vel, double * const y_vel,
            const double * const x_force, const double * const y_force,
            const double * const mass, double step) {

        *x_pos += (*x_vel) * step;
        *y_pos += (*y_vel) * step;

        double x_acc = *x_force / *mass;
        double y_acc = *y_force / *mass;

        *x_vel += x_acc * step;
        *y_vel += y_acc * step;

        /* compute statistics */
        double cur_acc = (x_acc * x_acc + y_acc * y_acc);
        cur_acc = sqrt(cur_acc);
        double speed_sq = (*x_vel) * (*x_vel) + (*y_vel) * (*y_vel);
        double cur_speed = sqrt(speed_sq);

        sum_speed_sq += speed_sq;
        max_acc = MAX(max_acc, cur_acc);
        max_speed = MAX(max_speed, cur_speed);
    }


#ifdef SOA
/*
 * Calculates the forces for all
 * the particles in the system
 */
    void nbody::calculate_forces() {
        /* First calculate force for particle_soa. */

        double *x_pos = particles_soa.x_pos, *y_pos = particles_soa.y_pos;
        double *x_vel = particles_soa.x_vel, *y_vel = particles_soa.y_vel;
        double *x_force = particles_soa.x_force, *y_force = particles_soa.y_force;
        double *mass = particles_soa.mass;
        int i=0, j=0;

        for (i=0; i< number_particles; i++) {
            x_force[i] = 0;
            y_force[i] = 0;
        }

        for (i = 0; i < number_particles; i++) {
            for (j = 0; j < number_particles; j++) {
                double x_sep, y_sep, dist_sq, grav_base;

                x_sep = x_pos[j] - x_pos[i];
                y_sep = y_pos[j] - y_pos[i];
                dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

                // Use the 2-dimensional gravity rule: F = d * (GMm/d^2)
                grav_base = GRAV_CONSTANT * (mass[i]) * (mass[j]) / dist_sq;

                x_force[i] += grav_base * x_sep;
                y_force[i] += grav_base * y_sep;
            }
        }
    }
#else
    void nbody::calculate_forces() {
        /* First calculate force for particle_soa. */

        for (int i = 0; i < number_particles; i++) {

            particles[i].x_force = 0;
            particles[i].y_force = 0;
            for (int j = 0; j < number_particles; j++) {
                particle_t *p = &particles[j];
                /* compute the force of particle j on particle i */
                compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
            }
        }
    }
#endif

#ifdef SOA
/*
 * Move particle_soa one time step.
 * Update positions, velocity, and acceleration.
 * Return local computations.
 */
    void nbody::move_all_particles(double step) {

        double *x_pos = particles_soa.x_pos, *y_pos = particles_soa.y_pos;
        double *x_vel = particles_soa.x_vel, *y_vel = particles_soa.y_vel;
        double *x_force = particles_soa.x_force, *y_force = particles_soa.y_force;
        double *mass = particles_soa.mass;

    /* then move all particle_soa and return statistics */
        for (int i = 0; i < number_particles; i++) {
            x_pos[i] += (x_vel[i]) * step;
            y_pos[i] += (y_vel[i]) * step;

            double x_acc = x_force[i] / mass[i];
            double y_acc = y_force[i] / mass[i];

            x_vel[i] += x_acc * step;
            y_vel[i] += y_acc * step;

            // compute statistics
            double cur_acc = (x_acc * x_acc + y_acc * y_acc);
            cur_acc = sqrt(cur_acc);
            double speed_sq = (x_vel[i]) * (x_vel[i])
                    + (y_vel[i]) * (y_vel[i]);
            double cur_speed = sqrt(speed_sq);

            sum_speed_sq += speed_sq;
            max_acc = MAX(max_acc, cur_acc);
            max_speed = MAX(max_speed, cur_speed);
        }
    }
#else
    void nbody::move_all_particles(double step) {

        /* then move all particle_soa and return statistics */
        for (int i = 0; i < number_particles; i++) {
            move_particle(&particles[i], step);
        }
    }
#endif

    void nbody::all_move_particles(double step) {
        calculate_forces();
        move_all_particles(step);
    }


/* display all the particle_soa */
    void nbody::draw_all_particles() {
#ifdef DISPLAY
        int i;
        for(i=0; i<number_particles; i++) {
          int x = POS_TO_SCREEN(particle_soa[i].x_pos);
          int y = POS_TO_SCREEN(particle_soa[i].y_pos);
          draw_point (x,y);
        }
#endif
    }

/*
  Place particle_soa in their initial positions.
*/

    void nbody::all_init_particles() {

        if (universe_seed)
            srand(universe_seed);

        if (universe == cadlabs::universe_t::ORIGINAL) {
            printf("Universe: original\n");
#ifdef SOA
            original(number_particles, particles_soa.mass,
                     particles_soa.x_pos, particles_soa.y_pos,
                     particles_soa.x_vel, particles_soa.y_vel);
#else
            original(number_particles, particles);
#endif
        } else if (universe == universe_t::DISC) {
            printf("Universe: disc\n");
#ifdef SOA
            rotating_disc(number_particles, particles_soa.mass,
                          particles_soa.x_pos, particles_soa.y_pos,
                          particles_soa.x_vel, particles_soa.x_vel);
#else
            rotating_disc(number_particles, particles);
#endif
        } else if (universe == universe_t::SPHERE) {
            printf("Universe: sphere\n");
#ifdef SOA
            sphere(number_particles, particles_soa.mass,
                   particles_soa.x_pos, particles_soa.y_pos);
#else
            sphere(number_particles, particles);
#endif
        }
    }

    void nbody::print_all_particles(std::ostream &out) {
        for (int i = 0; i < number_particles; i++) {
            particle_t *p = &particles[i];
            out << "particle={pos=(" << p->x_pos << "," << p->y_pos << "), vel=("
                << p->x_vel << "," << p->y_vel << ")}\n";
        }
    }

    void nbody::run_simulation() {
        double t = 0.0, dt = 0.01;
        while (t < T_FINAL && number_particles > 0) {
            /* Update time. */
            t += dt;
            /* Move particle_soa with the current and compute rms velocity. */
            all_move_particles(dt);

            /*
             * Adjust dt based on maximum speed and acceleration--this
             * simple rule tries to insure that no velocity will change
             * by more than 10%
            */

            dt = 0.1 * max_speed / max_acc;

            /* Plot the movement of the particle */
#if DISPLAY
            clear_display();
            draw_all_particles();
            flush_display();
#endif
        }
#ifdef DEBUG
#ifdef SOA
        debug->save_values_by_iteration(particles_soa.x_pos, particles_soa.y_pos,
                                        particles_soa.x_vel, particles_soa.y_vel, number_particles);
#else
        debug->save_values_by_iteration(particles, number_particles);
#endif
#endif
    }
}


