%nproc=4
%mem=5760MB
%chk=meoh_444.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4038 0.0854 -0.0428
C 0.0376 -0.0016 -0.0023
H 1.8777 -0.4536 0.6247
H -0.3422 0.4212 0.9278
H -0.4755 0.5070 -0.8185
H -0.3317 -1.0259 -0.0524

