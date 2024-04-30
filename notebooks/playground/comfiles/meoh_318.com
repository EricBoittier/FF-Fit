%nproc=4
%mem=5760MB
%chk=meoh_318.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4134 0.0429 0.0445
C 0.0242 -0.0127 0.0160
H 1.8246 0.2385 -0.8233
H -0.2720 0.3717 -0.9600
H -0.4160 -1.0069 0.0928
H -0.4006 0.6526 0.7677

