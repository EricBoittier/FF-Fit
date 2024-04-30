%nproc=4
%mem=5760MB
%chk=meoh_331.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4406 0.0824 0.0537
C 0.0062 0.0071 -0.0127
H 1.6440 -0.4483 -0.7447
H -0.4543 0.5148 -0.8603
H -0.2188 -1.0594 -0.0228
H -0.4523 0.3853 0.9010

