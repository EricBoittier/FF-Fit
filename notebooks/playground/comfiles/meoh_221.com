%nproc=4
%mem=5760MB
%chk=meoh_221.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4227 0.1040 -0.0088
C -0.0006 0.0040 -0.0016
H 1.8475 -0.7722 0.1024
H -0.2587 -0.7524 -0.7429
H -0.2621 -0.3663 0.9896
H -0.4425 0.9774 -0.2142

