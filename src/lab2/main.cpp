#include <stdio.h>
#include <stdlib.h>


#include "par_nbody_all_pairs.h"
#include <unistd.h>
#include <fstream>
#include <marrow/utils/timer.hpp>

void usage(char *prog) {
    fprintf(stderr, "usage: %s number_particles [-t duration time] [-u universe] [-s seed]\n"
                    "\t-t --> number of end time (default 1.0)\n"
                    "\t-n --> number of threads running the simulation (default number of hardware threads)\n"
                    "\t-u --> universe type [0 - line, 1 - sphere, 2 - rotating disc] (default 0)\n"
                    "\t-s --> seed for universe creation (if needed).\n"
                    "\t-# --> number of times running the simulation (default 1)\n", prog);
}

int main(int argc, char**argv) {

    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }

    auto nparticles = atoi(argv[1]);

    // default values
    float T_FINAL = 10.0;
    cadlabs::universe_t universe = cadlabs::universe_t::ORIGINAL;
    unsigned number_of_threads = 0; // default: automatic
    auto number_of_runs = 1;

    int c;
    while ((c = getopt(argc-1, argv+1, "t:u:s:#:")) != -1)
        switch (c) {
            case 't':
                T_FINAL = atof(optarg);
                break;

            case 'u':
                universe = static_cast<cadlabs::universe_t>(atoi(optarg));
                break;

            case 'n':
                number_of_threads = atoi(optarg);;
                break;

            case 's':
                //         universe_seed = atoi(optarg);;
                break;

            case '#':
                number_of_runs = atoi(optarg);;
                break;

            default:
                fprintf (stderr, "%c option not supported\n", c);
                usage(argv[0]);
                exit(1);
        }

    cadlabs::par_nbody_all_pairs nbody(nparticles, T_FINAL, universe, number_of_threads);
    marrow::timer<> t;

    for (int i = 0; i < number_of_runs; i++) {
        std::cout << "Simultion #" << i << "\n";
        t.start();
        nbody.run_simulation();
        t.stop();
        nbody.reset();
    }

#ifdef DUMP_RESULT
    std::ofstream myfile;
    myfile.open ("particles.log");
    nbody.print_all_particles(myfile);
    myfile.close();
#endif

    printf("-----------------------------\n");
    printf("nparticles: %d\n", nbody.number_particles);
    printf("T_FINAL: %f\n", nbody.T_FINAL);
    printf("-----------------------------\n");
    t.output_stats<false>(std::cout);

    return 0;
}