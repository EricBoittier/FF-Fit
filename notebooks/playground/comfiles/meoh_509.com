%nproc=4
%mem=5760MB
%chk=meoh_509.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4270 0.0439 0.0564
C 0.0343 0.0080 0.0006
H 1.6410 0.1297 -0.8962
H -0.4252 -0.5140 0.8400
H -0.4928 0.9616 -0.0300
H -0.3232 -0.5855 -0.8409

