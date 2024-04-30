%nproc=4
%mem=5760MB
%chk=meoh_937.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4414 0.0948 -0.0667
C -0.0157 -0.0276 0.0109
H 1.6604 -0.3243 0.7918
H -0.3414 0.9812 -0.2430
H -0.2842 -0.8123 -0.6963
H -0.2675 -0.2356 1.0508

