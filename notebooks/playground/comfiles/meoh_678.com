%nproc=4
%mem=5760MB
%chk=meoh_678.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4387 0.0038 0.0262
C -0.0058 -0.0002 0.0117
H 1.6974 0.7499 -0.5543
H -0.3021 -0.9920 -0.3298
H -0.3828 0.2140 1.0117
H -0.3220 0.7550 -0.7078

