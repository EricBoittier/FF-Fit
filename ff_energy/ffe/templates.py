import jinja2
from jinja2 import Template
from pathlib import Path

template_files = Path(__file__).parent / "template_files"

with open(template_files / "esp_view.sh") as file_:
    esp_view_template = Template(file_.read())

with open(template_files / "esp_cubes.sh") as file_:
    esp_cubes_template = Template(file_.read())


with open(template_files / "gaussian.com") as file_:
    g_template = Template(file_.read())
# print(g_template.render())

with open(template_files / "esp.vmd") as file_:
    vmd_template = Template(file_.read())


molpro_job_template = jinja2.Template(
    """***,Molpro Input
gprint,basis,orbitals=50,civector
gthresh,printci=0.0,energy=1.d-8,orbital=1.d-8,grid=1.d-8
!gdirect
symmetry,nosym
orient,noorient

memory,{{MEMORY}},m    ! memory: MEM_PER_CPU/8 * ~(0.90 to 0.95)
symmetry,nosym
orient,noorient
geometry={
angstrom;
{{XYZ}}
}
charge={{CHARGE}}
basis={{BASIS}}
{{RUN}}

edm=energy
name='{{NAME}}.molden'
!  save xyz file
TEXT,$name
PUT,molden,$name,NEW
!force
!expec,type=relax,dm
"""
)

m_slurm_template = jinja2.Template(
    """#!/bin/bash
#SBATCH --job-name={{NAME}}
##SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks={{NPROC}}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
##SBATCH --exclude=node[01-04,09-10]

hostname
echo $PWD

. /software/modules-4.8.0/init/sh

#---------
# Modules
#---------

# PC-Beethoven
#module molpro/molpro2022-gcc-9.2.0
# PC-Bach
#module load molpro/molpro-2022.1
# PC-NCCR
#module load molpro/molpro-mpp-2022.1

{ # try

    module molpro/molpro2022-gcc-9.2.0
    #save your output

} || { # catch
   echo "module molpro/molpro2022-gcc-9.2.0 (pc-beethoven) not available" # save log for exception
}

{ # try
    module load molpro/molpro-mpp-2022.1
    #save your output
} || { # catch
   echo "module load molpro/molpro-mpp-2022.1 (nccr) not available" # save log for exception
}


{ # try

    module load molpro/molpro-2022.1
    #save your output

} || { # catch
   echo "module load molpro/molpro-2022.1 (bach) not available" # save log for exception
}


# Get available memory from /tmp
tmp_mem=$(df -k /tmp | awk '{print $4}' | tail -1)
# Get available memory from /scratch
scratch_mem=$(df -k /scratch | awk '{print $4}' | tail -1)
# Compare the two and print out a message
if [ $tmp_mem -gt $scratch_mem ]
then
    echo "Available memory in /tmp is greater than /scratch"
    datadir=/tmp
elif [ $tmp_mem -lt $scratch_mem ]
then
    echo "Available memory in /scratch is greater than /tmp"
    datadir=/scratch
else
    echo "Available memory in /tmp is equal to /scratch"
    datadir=/scratch
fi

#------------
# Parameters
#------------
# datadir=/scratch
export MAINDIR=$(pwd)

# Molpro file (expect input file with .inp suffix, but don't put here, see below)
export MOLFILE={{NAME}}  #your_molpro_file

export INPFILE=$MOLFILE.inp
export OUTFILE=$MOLFILE.out

# Get active job IDs
active_jobs=$(squeue -h -r -u $USER -o "%i")
# Loop through tmp directories
for dir in $datadir/$USER/jobs.py/*
do
   if [ -d "$dir" ]; then
  # Extract job ID from directory name
  job_id=$(basename -- $dir | sed 's/tmp.//')

  # Check if job ID is active
  if [[ $active_jobs =~ $job_id ]]
  then
    echo "Job $job_id is active, keeping directory $dir"
  else
    echo "Job $job_id is inactive, removing directory $dir"
    rm -rf $dir
  fi
    fi
done

# Get the available disk space
AVAILABLE_DISK_SPACE=$(df -k $datadir | awk 'FNR == 2 {print $4}')

echo "Available disk space: " $AVAILABLE_DISK_SPACE

# If the available disk space is lower than 200 GB, sleep
while [ $AVAILABLE_DISK_SPACE -lt 207200000 ];
do
        echo $AVAILABLE_DISK_SPACE 'sleeping'
        sleep 1m
        AVAILABLE_DISK_SPACE=$(df -k $datadir | awk 'FNR == 2 {print $4}')
done


# Make temporary directory
export TMPDIR=$datadir/$USER/jobs.py/tmp.$SLURM_JOBID
if [ -d $TMPDIR ]; then
  echo "$TMPDIR exists; double job start; exit"
  exit 1
fi
mkdir -p $TMPDIR

#-------------
# Computation
#-------------

# Run MOLPRO
molpro -d $TMPDIR -I $TMPDIR -W $TMPDIR -o $TMPDIR/$OUTFILE --no-xml-output $MAINDIR/$INPFILE

# Copy result file to output directory
cp $TMPDIR/$OUTFILE $MAINDIR/$OUTFILE
cp $TMPDIR/$MOLFILE.molden $MAINDIR/$MOLFILE.molden

# Remove temporary directory
rm -rf $TMPDIR

"""
)


orbkit_ci_template = jinja2.Template(
    r"""import sys

sys.path.append("/home/boittier/orbkit-testsuite/orbkit")
sys.path.append("/opt/cluster/programs/libcint/lib64")

from orbkit import read, grid, extras, output, display, analytical_integrals, libcint_interface

tokcal=627.503

# Start by combining 2 monomer wfns to a frozen-density supermolecule wfn
qc = read.main_read('{{M1}}',itype='molden',all_mo=False)
qc2 = read.read_monomer2('{{M2}}',qc,itype='molden',all_mo=False)
ao = libcint_interface.AOIntegrals(qc2)

print('EDA Eel RESULTS:')

# MO Nuclear-Electron between frozen monomers
Vne=[]
Vne = ao.VneEDA(asMO=True)
Vnetot=0
for i in range(len(Vne)):
  #E+=qc2.mo_spec[i]['occ_num']*Vne[i,i]
  Vnetot+=2.0*Vne[i]
print('E(Nuc-elec): ',Vnetot)

# Nuclear-Nuclear
Vnn = qc2.EDA_nuclear_repulsion
print('E(Nuc-Nuc) = ',str(Vnn))

# MO Electron-Electron
Vee = ao.VeeEDA(asMO=True)
Veetot = 0.0
for i in range(len(Vee)):
  for j in range(len(Vee[i])):
    #Veetot += qc2.mo_spec[i]['occ_num']*qc2.mo_spec[j]['occ_num']*Vee[i][j]
    Veetot += 2.0*Vee[i][j]
print('E(elec-elec) = ',str(Veetot))

Eel=Vnetot+Vnn+Veetot
print('FINAL EEL RESULT:\n',str(Eel),' Hartree')

print(str(Eel*tokcal),' kcal/mol')

"""
)

o_slurm_template = jinja2.Template(
    """#!/bin/bash
#SBATCH --job-name={{NAME}}
##SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10200

hostname
{{COMMAND}}
"""
)

c_slurm_template = jinja2.Template(
    """#!/bin/bash
#SBATCH --job-name={{NAME}}
##SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1200

{{MODULES}}
charmm={{CHARMMPATH}}
{{COMMAND}}
"""
)

PSF = jinja2.Template(
    """
read rtf card
* methanol
*
31 1

MASS     5  HT        1.00800 H ! TIPS3P WATER HYDROGEN
MASS    75  OT       15.99940 O ! TIPS3P WATER OXYGEN
MASS  1  CG331     12.01100 ! aliphatic C for methyl group (-CH3)
MASS  2  HGP1       1.00800 ! polar H
MASS  3  HGA3       1.00800 ! alphatic proton, CH3
MASS  4  OG311     15.99940 ! hydroxyl oxygen
MASS  -1  CG321     12.01100 ! aliphatic C for CH2
MASS  -1  CLGA1     35.45300 ! CLET, DCLE, chloroethane, 1,1-dichloroethane
MASS  -1  HGA2       1.00800 ! alphatic proton, CH2
MASS -1 CLA       35.45000 ! CLET, DCLE, chloroethane, 1,1-dichloroethane
MASS -1 POT      39.09830  ! potassium ion

DEFA FIRS NONE LAST NONE
AUTO ANGLES DIHE PATCH DRUDE

RESI {{METHANOL}}            0.000 ! param penalty=   0.000 ; charge penalty=   0.000
GROUP            ! CHARGE   CH_PENALTY
ATOM {{OM}}      OG311  -0.650 !    0.000
ATOM {{CM}}      CG331  -0.040 !    0.000
ATOM {{H1M}}     HGP1    0.420 !    0.000
ATOM {{H2M}}     HGA3    0.090 !    0.000
ATOM {{H3M}}     HGA3    0.090 !    0.000
ATOM {{H4M}}     HGA3    0.090 !    0.000

BOND {{CM}}     {{OM}}
BOND {{OM}}    {{H1M}}
BOND {{CM}}     {{H4M}}
BOND {{CM}}     {{H3M}}
BOND {{CM}}     {{H2M}}
DONO {{H1M}}    {{OM}}
ACCE {{OM}}

PATC  FIRS NONE LAST NONE

RESI {{WATER}}         0.000 ! tip3p water model, generate using noangle nodihedral
GROUP
ATOM {{O}}    OT     -0.834
ATOM {{H}}    HT      0.417
ATOM {{H1}}   HT      0.417
BOND {{O}} {{H}} {{O}} {{H1}} {{H}} {{H1}}    ! the last bond is needed for shake
ANGLE {{H}} {{O}} {{H1}}             ! required
ACCEPTOR {{O}}
PATCHING FIRS NONE LAST NONE

RESI DCM       0.000 ! param penalty=   4.000 ; charge penalty=   0.000

GROUP            ! CHARGE   CH_PENALTY
ATOM C      CG321  -0.018 !    0.000
ATOM CL1    CLGA1  -0.081 !    0.000
ATOM CL2    CLGA1  -0.081 !    0.000
ATOM H1     HGA2    0.090 !    0.000
ATOM H2     HGA2    0.090 !    0.000

BOND H1   C
BOND C    CL1
BOND C    CL2
BOND C    H2
PATCHING FIRS NONE LAST NONE


RESI POT       1.00 ! Potassium Ion
GROUP
ATOM POT   POT 1.00
PATCHING FIRST NONE LAST NONE

RESI CLA      -1.00 ! Chloride Ion
GROUP
ATOM CLA  CLA -1.00
PATCHING FIRST NONE LAST NONE


END
"""
)

c_job_template = jinja2.Template(
    """* DCM water/ion params test for energy and forces
*

set base     /home/boittier/charmm/test
set pardir   @base
set crddir   @base

PRNLev 5

{{PSF}}

{{PAR}}

!================================================================
! Read coordinates
!================================================================
OPEN UNIT 1 READ FORM NAME {{PDB}}
READ SEQU PDB UNIT 1
CLOSE UNIT 1
GENERATE {{RES}} FIRST NONE LAST NONE SETUP NOANG NODIHED

OPEN UNIT 1 READ FORM NAME {{PDB}}
READ COOR PDB UNIT 1
CLOSE UNIT 1

!================================================================
! MDCM
!================================================================
open unit 15 write card name dcm.xyz
{{DCM_COMMAND}}

NBONd CUTNb 100.0 CTONnb 90.0 CTOFnb 94.0 E14FAC 0.0 FSWITch VSWItch CDIElectric EPSilon 1.0 NBXMOD 5

ENERGY
"""
)

PAR = """read parameter card
* methanol
*
ATOMS
MASS     5  HT        1.00800 ! TIPS3P WATER HYDROGEN
MASS    75  OT       15.99940 ! TIPS3P WATER OXYGEN
MASS  1  CG331     12.01100 ! aliphatic C for methyl group (-CH3)
MASS  2  HGP1       1.00800 ! polar H
MASS  3  HGA3       1.00800 ! alphatic proton, CH3
MASS  4  OG311     15.99940 ! hydroxyl oxygen
MASS  -1  CG321     12.01100 ! aliphatic C for CH2
MASS  -1  CLGA1     35.45300 ! CLET, DCLE, chloroethane, 1,1-dichloroethane
MASS  -1  HGA2       1.00800 ! alphatic proton, CH2
MASS -1 CLA       35.45000 ! CLET, DCLE, chloroethane, 1,1-dichloroethane
MASS -1 POT      39.09830  ! potassium ion

BONDS
CG331  OG311  428.0     1.420
CG331  HGA3   322.0     1.111
OG311  HGP1   545.0     0.960
OT  HT  450.0 0.9572 ! ALLOW WAT
HT  HT    0.0 1.5139 ! ALLOW WAT
CG321  CLGA1   220.00     1.7880 ! CLET, chloroethane
CG321  HGA2    309.00     1.1110 ! PROT alkane update, adm jr., 3/2/92

ANGLES
OG311  CG331 HGA3 45.90  108.8900
HGA3   CG331 HGA3 35.50  108.4000
CG331  OG311 HGP1 57.50  106.0000
HT  OT  HT   55.0 104.52 ! ALLOW WAT
OT  HT  HT   55.0 104.52 ! ALLOW WAT
CLGA1  CG321  CLGA1    95.00    109.00 ! dcm_freq , from CLGA1 CG311 CLGA1, PENALTY= 4
CLGA1  CG321  HGA2     42.00    107.00 ! CLET, chloroethane
HGA2   CG321  HGA2     35.50    109.00    5.40   1.80200 ! PROT alkane update, adm jr., 3/2/92

DIHEDRALS
HGA3 CG331 OG311 HGP1     0.18        3     0.0000

IMPROPERS

NONBONDED
OG311    0.0       -0.192   1.765                  ! og MeOH and EtOH 1/06 (was -0.1521 1.7682)
CG331    0.0       -0.078   2.050   0.0 -0.01 1.9 ! alkane (CT3), 4/98, yin, adm jr; Rmin/2 modified from 2.04 to 2.05
HGP1     0.0       -0.046    0.225                 ! polar H
HGA3     0.0       -0.024    1.340                 ! alkane, yin and mackerell, 4/98
OT     0.00  -0.1521  1.7682 ! ALLOW   WAT
HT     0.00  -0.0460  0.2245 ! ALLOW WAT
CG321    0.0       -0.0560     2.0100   0.0 -0.01 1.9 ! alkane (CT2), 4/98, yin, adm jr, also used by viv
CLGA1    0.0       -0.3430     1.9100 ! CLET, DCLE, chloroethane, 1,1-dichloroethane
HGA2     0.0       -0.0240     1.3400 ! alkane, yin and mackerell, 4/98
CLA      0.0       -0.150      2.27     ! Chloride
                   ! D. Beglovd and B. Roux, dA=-83.87+4.46 = -79.40 kcal/mol
POT      0.0       -0.0870    1.76375   ! Potassium
                   ! D. Beglovd and B. Roux, dA=-82.36+2.8 = -79.56 kca/mol
                   
END"""

PAR_f = """read parameter card
* methanol
*
ATOMS
MASS     5  HT        1.00800 ! TIPS3P WATER HYDROGEN
MASS    75  OT       15.99940 ! TIPS3P WATER OXYGEN
MASS  1  CG331     12.01100 ! aliphatic C for methyl group (-CH3)
MASS  2  HGP1       1.00800 ! polar H
MASS  3  HGA3       1.00800 ! alphatic proton, CH3
MASS  4  OG311     15.99940 ! hydroxyl oxygen
MASS  -1  CG321     12.01100 ! aliphatic C for CH2
MASS  -1  CLGA1     35.45300 ! CLET, DCLE, chloroethane, 1,1-dichloroethane
MASS  -1  HGA2       1.00800 ! alphatic proton, CH2
MASS -1 CLA       35.45000 ! CLET, DCLE, chloroethane, 1,1-dichloroethane
MASS -1 POT      39.09830  ! potassium ion

BONDS
CG331  OG311  428.0     1.420
CG331  HGA3   322.0     1.111
OG311  HGP1   545.0     0.960
CG321  CLGA1   220.00     1.7880 ! CLET, chloroethane
CG321  HGA2    309.00     1.1110 ! PROT alkane update, adm jr., 3/2/92
OT  HT  450.0 0.9572 ! ALLOW WAT
HT  HT    0.0 1.5139 ! ALLOW WAT

ANGLES
OG311  CG331 HGA3 45.90  108.8900
HGA3   CG331 HGA3 35.50  108.4000
CG331  OG311 HGP1 57.50  106.0000
HT  OT  HT   55.0 104.52 ! ALLOW WAT
OT  HT  HT   55.0 104.52 ! ALLOW WAT
CLGA1  CG321  CLGA1    95.00    109.00 ! dcm_freq , from CLGA1 CG311 CLGA1, PENALTY= 4
CLGA1  CG321  HGA2     42.00    107.00 ! CLET, chloroethane
HGA2   CG321  HGA2     35.50    109.00    5.40   1.80200 ! PROT alkane update, adm jr., 3/2/92


DIHEDRALS
HGA3 CG331 OG311 HGP1     0.18        3     0.0000

IMPROPERS

NONBONDED
OG311    0.0       {OG311_e}   {OG311_s}                  ! og MeOH and EtOH 1/06 (was -0.1521 1.7682)
CG331    0.0       {CG331_e}   {CG331_s}   0.0 -0.01 1.9 ! alkane (CT3), 4/98, yin, adm jr; Rmin/2 modified from 2.04 to 2.05
HGP1     0.0       {HGP1_e}    {HGP1_s}                 ! polar H
HGA3     0.0       {HGA3_e}   {HGA3_s}                 ! alkane, yin and mackerell, 4/98
OT       0.0    {OT_e}  {OT_s} ! ALLOW   WAT
HT       0.0    {HT_e}  {HT_s} ! ALLOW WAT
CG321    0.0       -0.0560     2.0100   0.0 -0.01 1.9 ! alkane (CT2), 4/98, yin, adm jr, also used by viv
CLGA1    0.0       -0.3430     1.9100 ! CLET, DCLE, chloroethane, 1,1-dichloroethane
HGA2     0.0       -0.0240     1.3400 ! alkane, yin and mackerell, 4/98
CLA      0.0       -0.150      2.27     ! Chloride
                   ! D. Beglovd and B. Roux, dA=-83.87+4.46 = -79.40 kcal/mol
POT      0.0       -0.0870    1.76375   ! Potassium
                   ! D. Beglovd and B. Roux, dA=-82.36+2.8 = -79.56 kca/mol
                   
END
"""

molpro_pol_template = jinja2.Template(
    """gprint,basis,orbitals=50,civector
gthresh,printci=0.0,energy=1.d-8,orbital=1.d-8,grid=1.d-8
symmetry,nosym
orient,noorient
angstrom
geomtyp=xyz
GEOMETRY={
{{XYZ}}
}

Lattice,Infile={{LAT}}

basis={{BASIS}}

{{RUN}}
edm=energy
name='{{JOBNAME}}_QMMM.molden'
TEXT,$name
PUT,molden,$name,NEW
"""
)

orbkit_pol_template = jinja2.Template(
    """import sys

sys.path.append("/home/boittier/orbkit-testsuite/orbkit")
sys.path.append("/opt/cluster/programs/libcint/lib64")

from orbkit import read, grid, extras, output, display, analytical_integrals, libcint_interface
import numpy as np

qc2 = read.main_read('{{MOLDEN}}',itype='molden',all_mo=False)
qc2.load_chgs('{{LAT}}')
ao = libcint_interface.AOIntegrals(qc2)

# MO KE integrals
T = ao.kinetic(asMO=True)
Ttot = 0.0
for i in range(len(T)):
  Ttot += 2.0*T[i,i]
print('T(elec) = ',str(Ttot))

# MO Nuclear-Electron
Vne = ao.Vne(asMO=True)
Vnetot = 0.0
for i in range(len(Vne)):
  Vnetot += 2.0*Vne[i,i]
print('E(Nuc-elec) = ',str(Vnetot))

# Total 1e energy:
Eone=Ttot + Vnetot
print('E(1-e) = T(elec)+E(Nuc-elec): ',Eone)

# MO Electron-Electron for occupied MOs
Nelec = int(abs(qc2.get_charge(nuclear=False)))
ao.add_MO_block_2e(MOrange=range(Nelec//2)) # only works for closed-shell (rounding)?
Vee = ao.Vee(asMO=True, max_dims=0)
Veetot = 0.0
J=0.0
K=0.0

for i in range(len(Vee)):
  for j in range(len(Vee)):
    J += Vee[i,i,j,j]
    K += Vee[j,i,j,i]
print('J = ',str(J))
print('K = ',str(K))

Veetot = 2.0*J - K
print('E(elec-elec) = 2*J-K = ',str(Veetot))

# Nuclear-Nuclear
Vnn = qc2.nuclear_repulsion
print('E(Nuc-Nuc) = ',str(Vnn))

print('Total energy: E(1-e))+E(elec-elec)+E(Nuc-Nuc):')
Vtot = Eone + Veetot + Vnn
print('Total energy = ',str(Vtot))


#MO Energies
for i in range(len(Vne)):
  EMO = Vne[i,i] + T[i,i]
  tt=EMO
  for j in range(len(Vee)):
    EMO += 2.0*Vee[i,i,j,j] - Vee[j,i,j,i]
  print('MO ',str(i+1),' = ',str(EMO))

print('NOW CALCULATING INTERACTION WITH POINT CHARGES')

Vnnmm = qc2.QMMM_nuclear_repulsion
print('Energy (Nuc-QM/MM):',str(Vnnmm))

Vemm = ao.Vemm(asMO=True)
Vemmtot = 0.0
for i in range(len(Vemm)):
  Vemmtot += 2.0*Vemm[i,i]
print('Energy (elec-QM/MM):',str(Vemmtot))

#MO Energies
for i in range(len(Vemm)):
  EMO = Vemm[i,i] + Vne[i,i] + T[i,i]
  tt=EMO
  for j in range(len(Vee)):
    EMO += 2.0*Vee[i,i,j,j] - Vee[j,i,j,i]
  print('MO ',str(i+1),' = ',str(EMO))

VTotmm = (Vnnmm + Vemmtot)
print('Total interaction with charges: ',str(VTotmm))

Tot = Vtot + VTotmm
print('TOTAL QM/MM SYSTEM ENERGY: ',str(Tot))
"""
)
