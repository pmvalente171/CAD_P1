#!/bin/bash

#
# usage:  run_experiments.sh build_directory
#

# Number of thread configurations tested
nthreads=(1 2 4 8 12 16 20 26 32)

# Problem size configurations tested
problem_size=(1000 1500 2000 3000 5000)


# Find the build directory
base_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ $# -ne 1 ]; then
  build_dir="build"
else
  build_dir=$1
fi

if [ ! -d "${base_dir}/${build_dir}" ]; then
  echo "error: directory ${base_dir}/${build_dir} does not exist"
  exit 1
fi

pushd ${base_dir}/${build_dir}

# Compile before running
echo "Compiling"
make

# Run sequential version
for s in "${problem_size[@]}"
do
  command="src/lab1/nbody $s -t 10 -u 0 -# 4"
	echo "Running ${command}"
	${command}
done

# Run parallel version
for t in "${nthreads[@]}"
do
  for s in "${problem_size[@]}"
  do
    command="src/lab2/par_nbody $s -t 10 -u 0 -n $t -# 4"
    echo "Running ${command}"
    ${command}
  done
done

popd