%nproc=4
%mem=5760MB
%chk=meoh_130.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4426 -0.0017 0.0425
C -0.0095 0.0087 -0.0127
H 1.6823 0.7381 -0.5539
H -0.2913 -1.0332 -0.1651
H -0.3429 0.3361 0.9720
H -0.3733 0.6679 -0.8009

