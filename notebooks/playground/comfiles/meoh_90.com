%nproc=4
%mem=5760MB
%chk=meoh_90.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4313 -0.0105 -0.0014
C 0.0087 0.0016 -0.0002
H 1.7211 0.9257 0.0017
H -0.3673 -1.0215 0.0004
H -0.3579 0.5148 0.8888
H -0.3590 0.5138 -0.8893

