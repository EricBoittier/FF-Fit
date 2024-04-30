%nproc=4
%mem=5760MB
%chk=meoh_123.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4217 0.0007 0.0285
C 0.0126 0.0005 -0.0063
H 1.7995 0.7928 -0.4078
H -0.3467 -1.0213 -0.1288
H -0.3386 0.3791 0.9536
H -0.3713 0.6173 -0.8189

