#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <sys/time.h>

#include "par_nbody_all_pairs.h"
#include <unistd.h>

void usage(char *prog) {
    fprintf(stderr, "usage: %s number_particles [-t duration time] [-u universe] [-s seed]\n"
                    "\t-t --> number of end time (default 1.0)\n"
                    "\t-n --> number of threads running the simulation (default number of hardware threads)\n"
                    "\t-u --> universe type [0 - line, 1 - sphere, 2 - rotating disc] (default 0)\n"
                    "\t-s --> seed for universe creation. Used in disc.", prog);
}

int main(int argc, char**argv) {

    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }

    auto nparticles = atoi(argv[1]);

    // default values
    float T_FINAL = 10.0;
    universe_t universe = universe_t::ORIGINAL;
    unsigned number_of_threads = 0; // default: automatic

    int c;
    while ((c = getopt(argc-1, argv+1, "t:u:s:")) != -1)
        switch (c) {
            case 't':
                T_FINAL = atof(optarg);
                break;

            case 'u':
                universe = static_cast<universe_t>(atoi(optarg));
                break;

            case 'n':
                number_of_threads = atoi(optarg);;
                break;

            case 's':
                //         universe_seed = atoi(optarg);;
                break;

            default:
                fprintf (stderr, "%c option not supported\n", c);
                usage(argv[0]);
                exit(1);
        }

    cadlabs::par_nbody_all_pairs nbody(nparticles, T_FINAL, universe, number_of_threads);

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    /* Main thread starts simulation ... */
    nbody.run_simulation();

    gettimeofday(&t2, NULL);

    double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

#ifdef DUMP_RESULT
    FILE* f_out = fopen("particles.log", "w");
  assert(f_out);
  nbody.print_all_particles(f_out);
  fclose(f_out);
#endif

    printf("-----------------------------\n");
    printf("nparticles: %d\n", nbody.number_particles);
    printf("T_FINAL: %f\n", nbody.T_FINAL);
    printf("-----------------------------\n");
    printf("Simulation took %lf s to complete\n", duration);

    return 0;
}
