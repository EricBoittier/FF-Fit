%nproc=4
%mem=5760MB
%chk=meoh_153.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4347 0.0320 0.0593
C -0.0048 -0.0025 -0.0098
H 1.7145 0.3293 -0.8317
H -0.2946 -1.0048 -0.3255
H -0.3519 0.1995 1.0035
H -0.3236 0.7825 -0.6956

