# CAD course - NBody, implementations

## Project Structure

This project is composed of three main folders:

- **SEQ (src/SEQ)** - This folder contains the sequential implementations of our project, this includes one that uses an AoS (array of structures) and one using SoA (structure of arrays);
- **PAR (src/PAR)** - This folder contains a parallel implementation of the nbody problem using the *OpenMP* framework;
- **CUDA (src/CUDA)** - This folder contains several implementations of the nbody problem using CUDA, these are the following, it's important to note that every algorithm that follows has both a implementation using AoS and another using SoA : 
    - *1:* A first naive implementation that parallelizes the outer loop of the problem;
    - *2:* A solution that parallelizes both of the loops, but only uses global memory accesses, to preform reads and writes;
    - *3:* A solution that parallelizes both of the loops, but relies on the usage of shared memory, to preform temporary reads and writes;
    - *4:* A solution that uses the Map and reduce parallel patterns to implement the algorithm;

## How to run

In order for the programmer to switch between using a SoA data layout or an AoS layout, they have to go into the `include/nbody` folder and to the `dataType.h` file and either add or remove the following code:

```c++
#define SOA
```

In order to run both solutions *2* and *3*, the programmer has to go to the the `include/nbody` folder and to the `dataType.h` file and either add or remove the following code:

```c++
#define ATOMIC
```

### Flags

While running you have access to the following flags: 

    - \t-t --> number of end time (default 1.0);
    - \t-u --> universe type [0 - line, 1 - sphere, 2 - rotating disc] (default 0);
    - \t-s --> seed for universe creation (if needed);
    - \t-# --> number of times running the simulation (default 1);
    - \t-d --> prints to a file the particle_soa positions to an output file named arg;
    - \t-n --> the algorthmic complexity of the algorithm; 
    - \t-h --> the block width of the current execution;
    - \t-h --> the height of the current execution;
    - \t-o --> the number of streams;
