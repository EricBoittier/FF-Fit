#!/bin/bash

#SBATCH --job-name={{KEY}}
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1600MB
##SBATCH --partition=vshort
#SBATCH --exclude=node[109-124]
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
export g16root=/opt/cluster/programs/g16-c.01
source $g16root/g16/bsd/g16.profile
export GAUSS_SCRDIR=""
# Execute Gaussian jobs.py
$g16root/g16/g16 $INPFILE $OUTFILE

# Write formchk file
$g16root/g16/formchk $CHKFILE $FCKFILE

# Extract density cube file
$g16root/g16/cubegen 0 Density $FCKFILE $DNSFILE -3 h

# Extract esp cube file
$g16root/g16/cubegen 0 Potential $FCKFILE $ESPFILE -3 h

NAME={{KEY}}
XYZFILE=dcm.xyz
BINDIR=/data/boittier/MDCM/bin/
# get the ESP from xyz charges
$BINDIR/pcubefit.x -generate -xyz $XYZFILE -esp $ESPFILE -dens $DNSFILE -v > ${NAME}.cubegen.log

# Examine quality of fitted charges by comparing newly fitted model and reference
# MEP
$BINDIR/pcubefit.x -v -analysis -esp $ESPFILE -esp2 {{NCHG}}charges.cube -dens $DNSFILE > analyze-cube-${NAME}.log

echo "/home/boittier/bin/vmd/vmd_LINUXAMD64 -e {{KEY}}.vmd -dispdev text" > render.sh
bash render.sh
