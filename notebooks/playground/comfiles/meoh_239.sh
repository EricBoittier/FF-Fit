#!/bin/bash

#SBATCH --job-name=meoh_239
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1600MB
#SBATCH --partition=vshort
#SBATCH --exclude=node[109-124]
#$ -S /bin/bash

# Define Parameters
hostname
export WORKDIR=/home/boittier/methanol_dih/input
export DATADIR=/home/boittier/methanol_dih/data
export INPFILE=meoh_239.com
export OUTFILE=meoh_239.out
export CHKFILE=meoh_239.chk
export FCKFILE=meoh_239.fchk
export DNSFILE=meoh_239.d.cube
export ESPFILE=meoh_239.p.cube

# Make scratch temporary directory
export STMPDIR=/scratch/$USER/jobs/tmp.$SLURM_JOBID
if [ -d $STMPDIR ]; then
  echo "$STMPDIR exists; double job start; exit"
  exit 1
fi
mkdir -p $STMPDIR

cd $STMPDIR

# Copy input file
cp $WORKDIR/$INPFILE $STMPDIR/$INPFILE

# Prepare Gaussian parameters
export g16root=/opt/cluster/programs/g16-c.01
source $g16root/g16/bsd/g16.profile
export GAUSS_SCRDIR=$STMPDIR

# Execute Gaussian jobs
$g16root/g16/g16 $STMPDIR/$INPFILE $STMPDIR/$OUTFILE
    
# Copy result file to output directory
cp $STMPDIR/$OUTFILE $DATADIR/$OUTFILE

# Delete output file if successfully copied
status=$?
if [ $status -eq 1 ]; then
    echo "$STMPDIR/$OUTFILE could not be transferred to $OUTPDIR/$OUTFILE"
fi
        
# Write formchk file
$g16root/g16/formchk $CHKFILE $FCKFILE

# Extract density cube file
$g16root/g16/cubegen 0 Density $FCKFILE $DNSFILE -2 h

# Extract esp cube file
$g16root/g16/cubegen 0 Potential $FCKFILE $ESPFILE -2 h

# Copy result to data directory
cp $CHKFILE $DNSFILE $ESPFILE $DATADIR/.

if [ $status -eq 0 ] 
then 
    # Delete temporary directory
    rm -rf $STMPDIR
else
    echo "$STMPDIR results could not be transferred to $DATADIR"
fi

