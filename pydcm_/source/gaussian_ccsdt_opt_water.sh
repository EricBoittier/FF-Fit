#!/bin/bash

#SBATCH --job-name=gaussian_ccsdt_opt_water
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=NONE
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=550
# 
hostname

 
#set -xv
export g16root=/opt/cluster/programs/g16-c.01
export GAUSS_SCRDIR=/scratch/toepfer/gaussian_ccsdt_opt_water
mkdir -p /scratch/toepfer/gaussian_ccsdt_opt_water
source $g16root/g16/bsd/g16.profile

$g16root/g16/g16 /home/toepfer/Project_FMDCM/Water/source/gaussian_ccsdt_opt_water.com /scratch/toepfer/gaussian_ccsdt_opt_water/gaussian_ccsdt_opt_water.out

# don't delete the result file if not able to copy to fileserver 
cp /scratch/toepfer/gaussian_ccsdt_opt_water/gaussian_ccsdt_opt_water.out /home/toepfer/Project_FMDCM/Water/source/gaussian_ccsdt_opt_water.out 
status=$?
if [ $status -eq 0 ] 
then 
   rm -rf /scratch/toepfer/gaussian_ccsdt_opt_water
else
   host=`/bin/hostname`
   /usr/bin/Mail -v -s "Error at end of batch job" $USER <<EOF

At the end of the batch job the system could not copy the output file
	$host:/scratch/toepfer/gaussian_ccsdt_opt_water/gaussian_ccsdt_opt_water.out
to
	/home/toepfer/Project_FMDCM/Water/source/gaussian_ccsdt_opt_water.out
Please copy this file by hand or inform the system manager.

EOF
 
fi
