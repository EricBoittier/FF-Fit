%nproc=4
%mem=5760MB
%chk=meoh_808.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4340 0.0842 -0.0717
C -0.0085 -0.0080 0.0294
H 1.7436 -0.3553 0.7478
H -0.2494 -0.3937 -0.9612
H -0.3054 -0.7103 0.8083
H -0.3905 1.0034 0.1683

