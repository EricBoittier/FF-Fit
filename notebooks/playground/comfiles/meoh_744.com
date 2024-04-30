%nproc=4
%mem=5760MB
%chk=meoh_744.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4324 0.0931 0.0334
C -0.0039 -0.0063 0.0169
H 1.7564 -0.5081 -0.6694
H -0.2283 -0.7786 -0.7189
H -0.3679 -0.2633 1.0116
H -0.3916 0.9318 -0.3802

