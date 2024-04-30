%nproc=4
%mem=5760MB
%chk=meoh_282.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4514 -0.0059 -0.0444
C -0.0120 0.0000 0.0139
H 1.5583 0.8147 0.4807
H -0.3349 -0.0869 -1.0235
H -0.3441 -0.8206 0.6499
H -0.3146 0.9707 0.4068

