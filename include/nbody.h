#ifndef NBODY_H
#define NBODY_H

#include <ostream>
#include "nbody/nbody_universe.h"
#include "get_output.h"
using namespace data_types;

namespace cadlabs {

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

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))  /* utility function */
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))  /* utility function */



    class nbody {

    public:

        /**
         * Create an NBody simulation
         * @param number_particles  Number of particle_soa
         * @param t_final simulation end time
         * @param universe universe type
         * @param universe_seed Seed for the creation of the universe (if needed). Default 0 (no seed)
         */
        nbody(const int number_particles, const float t_final, const universe_t universe, const unsigned universe_seed = 0, const string file_name = "");

        virtual ~nbody();

        /**
         * Run the simulation
         */
        void run_simulation();

        /**
         * Reset the particle_soa to their original positions
         */
        void reset();

        /**
         * Print the final position of the particle_soa in a given output stream
         * @param out the output stream
         */
        void print_all_particles(std::ostream &out);


        /**
         * Number of particle_soa
         */
        const int number_particles;

        /**
         * simulation end time
         */
        const float T_FINAL;

        /**
         * universe type
         */
        const universe_t universe;

        /**
         * Seed for the creation of the universe
         */
        const unsigned universe_seed;


    protected:

        void all_move_particles(double step);

        void draw_all_particles();

        void compute_force(particle_t *p, double x_pos, double y_pos, double mass);

        void compute_force(float * const x_pos, float * const y_pos,
                           float * const x_force, float * const y_force, float *const mass,
                           float other_x_pos, float other_y_pos, float other_mass);

        void move_particle(particle_t *p, double step);

        void move_particle(float *x_pos, float *y_pos,
                           float *x_vel, float *y_vel,
                           float *x_force, float *y_force,
                           float *mass, double step);

        void all_init_particles();

        virtual void move_all_particles(double step);

        virtual void calculate_forces();

        /**
         * The array of particles (aos)
         */
        particle_t *particles;

#ifdef SOA
        /**
         * The structure of arrays (soa)
         */
        particle_soa particles_soa = particle_soa();
#endif
        /**
         * The debug class
         */
         get_output *debug;

    };
}
#endif
