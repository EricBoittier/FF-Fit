%nproc=4
%mem=5760MB
%chk=meoh_136.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4351 0.0061 0.0447
C 0.0146 0.0033 -0.0019
H 1.6761 0.6560 -0.6481
H -0.3601 -0.9932 -0.2360
H -0.4378 0.3116 0.9406
H -0.3721 0.6749 -0.7683

