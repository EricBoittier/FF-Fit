%nproc=4
%mem=5760MB
%chk=meoh_342.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4182 0.1054 0.0110
C 0.0096 -0.0037 0.0081
H 1.8236 -0.7527 -0.2334
H -0.3562 0.6652 -0.7710
H -0.2677 -1.0148 -0.2901
H -0.3659 0.2577 0.9975

