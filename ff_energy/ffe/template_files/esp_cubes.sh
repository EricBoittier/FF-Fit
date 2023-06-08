#!/bin/bash
## currently set to work on PCBACH
#SBATCH --job-name={{KEY}}
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1600MB
##SBATCH --partition=vshort
##SBATCH --exclude=node[109-124]
#$ -S /bin/bash

# Define Parameters
hostname

export INPFILE={{KEY}}.com
export OUTFILE={{KEY}}.out
export CHKFILE={{KEY}}.chk
export FCKFILE={{KEY}}.fchk
export DNSFILE={{KEY}}_dens.cube
export ESPFILE={{KEY}}_esp.cube

# Prepare Gaussian parameters
#export g16root=/opt/cluster/programs/g16-c.01
export g16root=/cluster/software/g16-c.01

source $g16root/g16/bsd/g16.profile
export GAUSS_SCRDIR=""
# Execute Gaussian jobs.py
$g16root/g16/g16 $INPFILE $OUTFILE

# Write formchk file
$g16root/g16/formchk $CHKFILE $FCKFILE

# Extract density cube file
$g16root/g16/cubegen 0 Density $FCKFILE $DNSFILE -2 h

# Extract esp cube file
$g16root/g16/cubegen 0 Potential $FCKFILE $ESPFILE -2 h

