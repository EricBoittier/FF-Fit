%nproc=4
%mem=5760MB
%chk=meoh_777.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4170 0.1115 -0.0199
C 0.0218 -0.0051 0.0211
H 1.8111 -0.7833 0.0484
H -0.2491 -0.6239 -0.8344
H -0.3894 -0.4807 0.9115
H -0.4651 0.9642 -0.0862

