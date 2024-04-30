%nproc=4
%mem=5760MB
%chk=meoh_957.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4592 -0.0053 -0.0461
C -0.0152 0.0089 0.0080
H 1.5224 0.8124 0.4904
H -0.4001 1.0203 0.1388
H -0.3239 -0.5062 -0.9017
H -0.3201 -0.5631 0.8843

