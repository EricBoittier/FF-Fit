%nproc=4
%mem=5760MB
%chk=meoh_594.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4290 0.0593 -0.0659
C 0.0092 -0.0139 0.0017
H 1.7836 0.0498 0.8477
H -0.3285 -0.9982 0.3261
H -0.3116 0.6989 0.7614
H -0.4761 0.2582 -0.9355

