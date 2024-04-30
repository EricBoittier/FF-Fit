%nproc=4
%mem=5760MB
%chk=meoh_129.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4410 -0.0019 0.0411
C -0.0099 0.0083 -0.0130
H 1.6979 0.7483 -0.5348
H -0.2893 -1.0355 -0.1563
H -0.3323 0.3406 0.9738
H -0.3720 0.6633 -0.8054

