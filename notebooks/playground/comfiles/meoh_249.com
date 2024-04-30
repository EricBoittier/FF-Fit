%nproc=4
%mem=5760MB
%chk=meoh_249.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4103 0.0731 -0.0541
C 0.0315 -0.0044 0.0010
H 1.8468 -0.2668 0.7549
H -0.4027 -0.4935 -0.8710
H -0.3370 -0.5526 0.8680
H -0.4093 0.9900 0.0706

