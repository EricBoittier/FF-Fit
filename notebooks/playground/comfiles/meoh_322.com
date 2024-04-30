%nproc=4
%mem=5760MB
%chk=meoh_322.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4269 0.0562 0.0473
C -0.0093 -0.0076 0.0103
H 1.8374 0.0243 -0.8421
H -0.2504 0.4230 -0.9616
H -0.3026 -1.0564 0.0570
H -0.3633 0.5930 0.8482

