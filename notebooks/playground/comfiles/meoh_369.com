%nproc=4
%mem=5760MB
%chk=meoh_369.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4364 0.0384 -0.0627
C -0.0273 0.0125 0.0074
H 1.7891 0.1281 0.8472
H -0.4174 0.9754 -0.3224
H -0.2090 -0.8042 -0.6913
H -0.1786 -0.2728 1.0485

