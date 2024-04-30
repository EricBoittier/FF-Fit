%nproc=4
%mem=5760MB
%chk=meoh_217.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4285 0.1112 -0.0009
C -0.0112 -0.0092 0.0007
H 1.8135 -0.7899 -0.0155
H -0.2620 -0.7547 -0.7539
H -0.2657 -0.3113 1.0166
H -0.3676 0.9858 -0.2658

