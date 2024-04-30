%nproc=4
%mem=5760MB
%chk=meoh_947.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4070 0.0434 -0.0533
C 0.0337 -0.0102 0.0063
H 1.8768 0.3000 0.7676
H -0.4106 0.9691 -0.1718
H -0.4218 -0.6014 -0.7881
H -0.3202 -0.4494 0.9390

