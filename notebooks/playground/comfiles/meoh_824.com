%nproc=4
%mem=5760MB
%chk=meoh_824.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4357 0.0491 -0.0633
C -0.0052 -0.0050 -0.0061
H 1.7533 0.1273 0.8606
H -0.4673 -0.2587 -0.9602
H -0.2258 -0.7571 0.7513
H -0.3273 0.9536 0.4005

