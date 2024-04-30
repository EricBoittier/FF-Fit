%nproc=4
%mem=5760MB
%chk=meoh_175.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4339 0.0730 0.0520
C -0.0093 -0.0140 0.0044
H 1.7723 -0.1856 -0.8307
H -0.2818 -0.9381 -0.5054
H -0.4105 0.0493 1.0159
H -0.2703 0.8671 -0.5819

