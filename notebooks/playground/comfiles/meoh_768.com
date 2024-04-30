%nproc=4
%mem=5760MB
%chk=meoh_768.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4290 0.1123 0.0027
C 0.0068 -0.0044 0.0123
H 1.7749 -0.7855 -0.1835
H -0.2894 -0.6688 -0.7995
H -0.3065 -0.4398 0.9612
H -0.4820 0.9481 -0.1927

