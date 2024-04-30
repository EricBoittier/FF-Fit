%nproc=4
%mem=5760MB
%chk=meoh_80.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4313 -0.0105 -0.0013
C 0.0087 0.0016 -0.0001
H 1.7211 0.9257 0.0015
H -0.3673 -1.0215 0.0003
H -0.3580 0.5148 0.8888
H -0.3589 0.5138 -0.8892

