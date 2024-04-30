%nproc=4
%mem=5760MB
%chk=meoh_609.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4369 0.0233 -0.0722
C -0.0085 0.0040 0.0246
H 1.7012 0.4307 0.7791
H -0.2381 -1.0528 0.1605
H -0.3664 0.6334 0.8394
H -0.3441 0.3553 -0.9511

