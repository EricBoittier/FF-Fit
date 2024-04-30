%nproc=4
%mem=5760MB
%chk=meoh_490.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4212 0.0040 0.0220
C 0.0367 -0.0143 0.0060
H 1.7079 0.8120 -0.4527
H -0.4378 -0.2113 0.9674
H -0.3479 0.9705 -0.2591
H -0.4589 -0.6786 -0.7019

