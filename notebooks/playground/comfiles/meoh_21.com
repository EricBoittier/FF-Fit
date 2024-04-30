%nproc=4
%mem=5760MB
%chk=meoh_21.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4313 -0.0105 -0.0004
C 0.0087 0.0016 0.0000
H 1.7212 0.9257 0.0004
H -0.3672 -1.0215 0.0000
H -0.3585 0.5145 0.8890
H -0.3587 0.5141 -0.8891

