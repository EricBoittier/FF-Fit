%nproc=4
%mem=5760MB
%chk=meoh_960.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4573 -0.0075 -0.0340
C -0.0212 0.0033 0.0053
H 1.5518 0.8961 0.3335
H -0.3455 1.0272 0.1915
H -0.3256 -0.4509 -0.9375
H -0.3011 -0.6074 0.8636

