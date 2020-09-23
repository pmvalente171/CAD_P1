#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "nbody.h"
#include "nbody_tools.h"
#include "ui.h"

extern float T_FINAL;     /* simulation end time */
extern particle_t*particles;





/*
  Simulate the movement of nparticles particles.
*/
int main(int argc, char**argv)
{
    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }

    nparticles = atoi(argv[1]);

    init(argc, argv);

    /* Allocate global shared arrays for the particles data set. */
    particles = malloc(sizeof(particle_t)*nparticles);
    all_init_particles(nparticles, particles);

    /* Initialize thread data structures */
#ifdef DISPLAY
    /* Open an X window to display the particles */
    simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
#endif

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    /* Main thread starts simulation ... */
    run_simulation();

    gettimeofday(&t2, NULL);

    double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

#ifdef DUMP_RESULT
    FILE* f_out = fopen("particles.log", "w");
  assert(f_out);
  print_all_particles(f_out);
  fclose(f_out);
#endif

    printf("-----------------------------\n");
    printf("nparticles: %d\n", nparticles);
    printf("T_FINAL: %f\n", T_FINAL);
    printf("-----------------------------\n");
    printf("Simulation took %lf s to complete\n", duration);

#ifdef DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();

    printf("Hit return to close the window.");

    getchar();
    /* Close the X window used to display the particles */
    XCloseDisplay(theDisplay);
#endif
    return 0;
}
