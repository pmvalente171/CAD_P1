h1. NBody simulation.

C++ implementation based on the C code available at http://www-inf.telecom-sudparis.eu/COURS/CSC5001/new_site/Supports/Projet/NBody/sujet.php
All new functionalities have a comment region with the developers name.

h2. Cmake configuration.

DISPLAY - Flag to configure the program to display the particles movement. Requires X11.
DUMP_RESULT - Flag to configure the program to dump the final positions of the particles to a file.
VECTORIZATION - Flag to set compiler's vectorization flags.

h2. Compile from the command line.

mkdir build
cd build
cmake ..
make

h2. Run the programs from the command line.

cd build
src/lab1/nbody -- runs the sequential all pairs implementation
src/lab2/par_nbody -- runs the parallel OpenMP-based all pairs implementation

h2. The run_experiments script.

The run_experiments script runs both the sequential and parallel versions varying the problem size and the number of threads.
