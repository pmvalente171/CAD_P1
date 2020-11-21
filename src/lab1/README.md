# NBody simulation.

C++ implementation based on the C code available at http://www-inf.telecom-sudparis.eu/COURS/CSC5001/new_site/Supports/Projet/NBody/sujet.php
All new functionalities have a comment region with the developers name.

## Cmake configuration.

- DISPLAY - Flag to configure the program to display the particle_soa movement. Requires X11.
- DUMP_RESULT - Flag to configure the program to dump the final positions of the particle_soa to a file.
- VECTORIZATION - Flag to set compiler's vectorization flags.

## Compile from the command line.

```
mkdir build
cd build
cmake ..
make
```

## Run the programs from the command line.

```
cd build
src/lab1/nbody -- runs the sequential all pairs implementation
```
