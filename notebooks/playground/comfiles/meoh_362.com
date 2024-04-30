%nproc=4
%mem=5760MB
%chk=meoh_362.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4298 0.0689 -0.0626
C 0.0237 0.0141 0.0046
H 1.6900 -0.3068 0.8045
H -0.5788 0.8341 -0.3861
H -0.3294 -0.8622 -0.5390
H -0.3006 -0.1405 1.0337

