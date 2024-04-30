%nproc=4
%mem=5760MB
%chk=meoh_147.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4270 0.0282 0.0475
C 0.0010 -0.0105 0.0056
H 1.7797 0.4492 -0.7643
H -0.3378 -0.9914 -0.3280
H -0.3912 0.2601 0.9859
H -0.2536 0.7445 -0.7382

