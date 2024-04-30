%nproc=4
%mem=5760MB
%chk=meoh_205.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4237 0.1113 0.0195
C 0.0291 -0.0150 0.0035
H 1.7485 -0.7410 -0.3393
H -0.3545 -0.7883 -0.6621
H -0.4477 -0.1834 0.9691
H -0.4325 0.9098 -0.3427

