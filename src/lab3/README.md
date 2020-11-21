# Lab 3

## Card info
A program that queries the system for the characteristics of the NVIDIA GPUs installed.

## CUDA container pairwise addition of containers
A CUDA implementation of the pairwise addition of two containers (container1 and container2) that can be generally writen as:

```
for i = 0 to result.size
  result[i] = container1[i] + container2[i]
```

## NBody simulation.

Codebase for C++/CUDA implementation based on the C code available at http://www-inf.telecom-sudparis.eu/COURS/CSC5001/new_site/Supports/Projet/NBody/sujet.php
All new functionalities have a comment region with the developers name.

## Cmake configuration.

- DISPLAY - Flag to configure the program to display the particle_soa movement. Requires X11.
- DUMP_RESULT - Flag to configure the program to dump the final positions of the particle_soa to a file.

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
src/lab2/par_nbody -- runs the parallel OpenMP-based all pairs implementation
src/lab3/cuda_nbody -- runs the parallel CUDA-based all pairs implementation
```




