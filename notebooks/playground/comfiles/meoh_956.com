%nproc=4
%mem=5760MB
%chk=meoh_956.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4559 -0.0030 -0.0490
C -0.0089 0.0091 0.0086
H 1.5370 0.7778 0.5377
H -0.4185 1.0139 0.1127
H -0.3337 -0.5186 -0.8881
H -0.3289 -0.5499 0.8879

