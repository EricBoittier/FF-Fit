%nproc=4
%mem=5760MB
%chk=meoh_972.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4069 0.0180 0.0270
C 0.0375 -0.0260 -0.0012
H 1.8077 0.7914 -0.4221
H -0.3284 0.9564 0.2974
H -0.4258 -0.1527 -0.9796
H -0.3720 -0.7860 0.6643

