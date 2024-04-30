%nproc=4
%mem=5760MB
%chk=meoh_621.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4222 0.0181 -0.0486
C 0.0259 -0.0160 0.0014
H 1.7718 0.6615 0.6028
H -0.4483 -0.9912 0.1125
H -0.2916 0.5621 0.8691
H -0.4553 0.4567 -0.8549

