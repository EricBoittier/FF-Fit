%nproc=4
%mem=5760MB
%chk=meoh_781.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4289 0.1165 -0.0250
C -0.0006 -0.0139 0.0146
H 1.7726 -0.7833 0.1561
H -0.2635 -0.6125 -0.8576
H -0.3279 -0.5028 0.9322
H -0.3958 1.0008 -0.0324

