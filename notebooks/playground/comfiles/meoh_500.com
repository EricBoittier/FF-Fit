%nproc=4
%mem=5760MB
%chk=meoh_500.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4360 0.0227 0.0426
C -0.0213 0.0009 0.0065
H 1.7691 0.4758 -0.7600
H -0.2894 -0.4215 0.9749
H -0.3416 1.0290 -0.1620
H -0.2180 -0.6684 -0.8311

