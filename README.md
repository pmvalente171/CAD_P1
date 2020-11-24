# Labs of the CAD course from MIEI@DI-FCT-UNL

## Project Structure

This project is composed of three main folders:
- **SEQ (src/SEQ)** - This folder contains the sequential implementations of our project, this includes one that uses an AoS (array of structures) and one using SoA (structure of arrays);
- **PAR (src/PAR)** - This folder contains a parallel implementation of the nbody problem using the *OpenMP* framework;
- **CUDA (src/CUDA)** - This folder contains several implementations of the nbody problem using CUDA, these are the following, it's important to note that every algorithm that follows has both a implementation using AoS and another using SoA : 
    - A first naive implementation that parallelizes the outer loop of the problem;
    - A solution that parallelizes both of the loops, but only uses global memory accesses, to preform reads and writes;
    - A solution that parallelizes both of the loops, but relies on the usage of shared memory, to preform temporary reads and writes;
    
    