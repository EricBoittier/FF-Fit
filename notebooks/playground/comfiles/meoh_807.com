%nproc=4
%mem=5760MB
%chk=meoh_807.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4351 0.0857 -0.0708
C -0.0112 -0.0070 0.0286
H 1.7456 -0.3805 0.7334
H -0.2387 -0.4109 -0.9580
H -0.3007 -0.7046 0.8145
H -0.3925 1.0045 0.1687

