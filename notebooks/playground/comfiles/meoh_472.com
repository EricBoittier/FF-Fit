%nproc=4
%mem=5760MB
%chk=meoh_472.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4150 0.0002 -0.0270
C 0.0142 0.0050 0.0058
H 1.8487 0.7989 0.3396
H -0.2918 -0.0272 1.0514
H -0.4180 0.8457 -0.5368
H -0.3089 -0.8943 -0.5186

