%nproc=4
%mem=5760MB
%chk=meoh_492.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4196 0.0093 0.0255
C 0.0276 -0.0150 0.0085
H 1.7611 0.7492 -0.5190
H -0.4055 -0.2481 0.9813
H -0.3388 0.9789 -0.2483
H -0.4188 -0.6641 -0.7448

