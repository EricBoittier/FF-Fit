%nproc=4
%mem=5760MB
%chk=meoh_534.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4232 0.1060 0.0233
C 0.0352 -0.0001 0.0048
H 1.7038 -0.6925 -0.4707
H -0.3418 -0.7502 0.7000
H -0.5372 0.8908 0.2629
H -0.3750 -0.3446 -0.9445

