%nproc=4
%mem=5760MB
%chk=meoh_323.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4313 0.0593 0.0489
C -0.0161 -0.0059 0.0073
H 1.8224 -0.0308 -0.8452
H -0.2564 0.4370 -0.9593
H -0.2747 -1.0638 0.0517
H -0.3599 0.5728 0.8646

