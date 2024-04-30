%nproc=4
%mem=5760MB
%chk=meoh_667.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4198 -0.0024 0.0112
C 0.0250 0.0020 0.0110
H 1.7489 0.8490 -0.3456
H -0.3416 -0.9939 -0.2380
H -0.4003 0.2566 0.9818
H -0.3820 0.6870 -0.7329

