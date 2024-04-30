%nproc=4
%mem=5760MB
%chk=meoh_219.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4259 0.1073 -0.0048
C -0.0087 -0.0022 -0.0006
H 1.8374 -0.7809 0.0431
H -0.2532 -0.7558 -0.7493
H -0.2551 -0.3409 1.0057
H -0.3992 0.9864 -0.2419

