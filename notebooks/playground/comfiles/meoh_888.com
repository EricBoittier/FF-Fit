%nproc=4
%mem=5760MB
%chk=meoh_888.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4319 0.0358 0.0552
C 0.0104 -0.0046 -0.0083
H 1.7177 0.2830 -0.8491
H -0.4310 0.5238 -0.8533
H -0.2972 -1.0495 0.0336
H -0.3821 0.5143 0.8663

