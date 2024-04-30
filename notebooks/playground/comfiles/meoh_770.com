%nproc=4
%mem=5760MB
%chk=meoh_770.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4217 0.1104 -0.0031
C 0.0186 -0.0013 0.0171
H 1.7980 -0.7851 -0.1336
H -0.2817 -0.6500 -0.8059
H -0.3441 -0.4526 0.9406
H -0.5015 0.9361 -0.1802

