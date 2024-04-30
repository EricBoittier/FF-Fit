%nproc=4
%mem=5760MB
%chk=meoh_695.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4324 0.0268 0.0444
C -0.0083 -0.0063 0.0019
H 1.7637 0.4773 -0.7605
H -0.3156 -0.9639 -0.4185
H -0.2929 0.0951 1.0492
H -0.3336 0.8263 -0.6219

