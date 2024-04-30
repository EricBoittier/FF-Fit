%nproc=4
%mem=5760MB
%chk=meoh_126.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4321 -0.0009 0.0354
C -0.0025 0.0051 -0.0111
H 1.7545 0.7727 -0.4726
H -0.3061 -1.0340 -0.1381
H -0.3212 0.3574 0.9699
H -0.3707 0.6429 -0.8147

