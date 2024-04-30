%nproc=4
%mem=5760MB
%chk=meoh_284.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4510 -0.0092 -0.0404
C -0.0093 0.0024 0.0150
H 1.5611 0.8532 0.4120
H -0.3364 -0.0700 -1.0222
H -0.3503 -0.8378 0.6198
H -0.3361 0.9573 0.4266

