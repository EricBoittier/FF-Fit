#!/bin/bash
#SBATCH --job-name={{JOBNAME}}
#SBATCH --nodes=1
#SBATCH --ntasks=12
##SBATCH --partition=infinite
##SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2200


hostname

nproc=12

module load gcc/gcc-12.2.0-cmake-3.25.1-openmpi-4.1.4
#module load charmm/c45a1-gcc9.2.0-ompi4.0.2
CHARMM=~/dev-release-dcm/build/cmake/charmm

mpirun -np $nproc $CHARMM -i dynamics.inp -o dynamics.log
#mpirun -np $nproc $CHARMM -i dynamics_restart_nve.inp -o dynamics.log
# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0