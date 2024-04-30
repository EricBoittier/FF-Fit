%nproc=4
%mem=5760MB
%chk=meoh_805.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4349 0.0882 -0.0675
C -0.0115 -0.0048 0.0250
H 1.7571 -0.4279 0.7008
H -0.2389 -0.4420 -0.9473
H -0.2937 -0.6883 0.8258
H -0.4046 1.0014 0.1696

